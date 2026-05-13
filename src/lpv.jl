## Linear Parameter-Varying (LPV) state-space identification.
#
# The user supplies a scheduling-variable trajectory λ(t) and a basis {φ_k} in λ.
# The estimated model has matrices
#
#     A(λ) = Σ_k θ^A_k φ_k(λ),    similarly for B(λ), C(λ), D(λ),
#
# i.e. a single shared state-space realization whose entries depend smoothly on λ.
# Estimation uses the same prediction-error scaffolding (Optim + ForwardDiff) as
# `structured_pem`, but with a hand-rolled LTV loop because the predictor is time-varying.

# ----------------------------------------------------------------------------
# Basis helpers
# ----------------------------------------------------------------------------

_normalize_basis(basis::Function, λ_probe) =
    (basis, length(basis(λ_probe)))

function _normalize_basis(basis::AbstractVector, λ_probe)
    isempty(basis) && throw(ArgumentError("basis must be non-empty"))
    fns = Tuple(basis)
    fn = let fns = fns
        λ -> [f(λ) for f in fns]
    end
    (fn, length(basis))
end

# Hand-rolled because the inner BFGS loop is differentiated through with ForwardDiff,
# and tensor-array operations (tullio/einsum/NNlib) drag in heavier dependencies.
function _contract(M::AbstractArray{<:Any,3}, φ)
    out = M[:, :, 1] .* φ[1]
    @inbounds for k in 2:length(φ)
        out .= out .+ M[:, :, k] .* φ[k]
    end
    out
end

# ----------------------------------------------------------------------------
# LPVStateSpace
# ----------------------------------------------------------------------------

"""
    LPVStateSpace{Tθ,Tb,TK,TT}

State-space model with matrices `A,B,C,D` parametrized as a basis expansion in a
scalar scheduling variable `λ`:

    A(λ) = Σ_k θ.A[:,:,k] * φ_k(λ)
    B(λ) = Σ_k θ.B[:,:,k] * φ_k(λ)
    C(λ) = Σ_k θ.C[:,:,k] * φ_k(λ)
    D(λ) = Σ_k θ.D[:,:,k] * φ_k(λ)

Fields:
- `basis::Tb`: callable `λ -> Vector` returning the `nb` basis values at `λ`
- `nb::Int`: number of basis functions
- `θ::Tθ`: `ComponentArray` with 3-D fields `A,B,C,D` (last axis is the basis dimension)
- `K::TK`: constant Kalman gain (the λ-varying case is left for future work)
- `Ts`: sample time
- `nx,nu,ny`: state, input, and output dimensions

Call `sys(λ)` to obtain a frozen `StateSpace` at a particular operating point.
See also [`lpv_pem`](@ref), [`lpv_warmstart`](@ref).

!!! warning "Experimental"
    LPV identification in this package is considered experimental and may
    change in the future without respecting semantic versioning. In particular,
    the field layout of [`LPVStateSpace`](@ref), the basis API, and the
    handling of the Kalman gain are subject to revision.
"""
struct LPVStateSpace{Tθ,Tb,TK,TT}
    basis::Tb
    nb::Int
    θ::Tθ
    K::TK
    Ts::TT
    nx::Int
    nu::Int
    ny::Int
end

function Base.show(io::IO, s::LPVStateSpace)
    print(io, "LPVStateSpace with nx=$(s.nx), nu=$(s.nu), ny=$(s.ny), nb=$(s.nb), Ts=$(s.Ts)")
end

function (s::LPVStateSpace)(λ)
    φ = s.basis(λ)
    A = _contract(s.θ.A, φ)
    B = _contract(s.θ.B, φ)
    C = _contract(s.θ.C, φ)
    D = _contract(s.θ.D, φ)
    ss(A, B, C, D, s.Ts)
end

ControlSystemsBase.ninputs(s::LPVStateSpace) = s.nu
ControlSystemsBase.noutputs(s::LPVStateSpace) = s.ny
ControlSystemsBase.nstates(s::LPVStateSpace) = s.nx

# ----------------------------------------------------------------------------
# LTV simulate/predict
# ----------------------------------------------------------------------------

"""
    simulate(sys::LPVStateSpace, u, λ; x0 = zeros(sys.nx))

Simulate the LPV system along the scheduling trajectory `λ`. `u` and `λ` must have
the same length along the time dimension.
"""
function simulate(sys::LPVStateSpace, u::AbstractMatrix, λ::AbstractVector; x0 = zeros(sys.nx))
    size(u, 2) == length(λ) || throw(ArgumentError("size(u, 2) must equal length(λ)"))
    T = promote_type(eltype(sys.θ), eltype(u), eltype(x0))
    N = size(u, 2)
    y = zeros(T, sys.ny, N)
    x = T.(copy(x0))
    @inbounds for t in 1:N
        φ = sys.basis(λ[t])
        At = _contract(sys.θ.A, φ)
        Bt = _contract(sys.θ.B, φ)
        Ct = _contract(sys.θ.C, φ)
        Dt = _contract(sys.θ.D, φ)
        ut = @view u[:, t]
        y[:, t] = Ct * x + Dt * ut
        x = At * x + Bt * ut
    end
    y
end

simulate(sys::LPVStateSpace, d::AbstractIdData, λ::AbstractVector; x0 = zeros(sys.nx)) =
    simulate(sys, time2(input(d)), λ; x0)

"""
    predict(sys::LPVStateSpace, d, λ; x0 = zeros(sys.nx), h = 1)

One-step-ahead prediction of the LPV system along the scheduling trajectory `λ`,
using the Kalman gain stored in `sys.K`. The innovation update reuses the
constant gain at every `t` — λ-varying `K` is not yet supported.
"""
function predict(sys::LPVStateSpace, d::AbstractIdData, λ::AbstractVector;
                 x0 = zeros(sys.nx), h::Int = 1)
    h == 1 || throw(ArgumentError("h > 1 not supported for LPVStateSpace yet"))
    length(d) == length(λ) || throw(ArgumentError("length(d) must equal length(λ)"))
    y = time2(output(d))
    u = time2(input(d))
    T = promote_type(eltype(sys.θ), eltype(y), eltype(u), eltype(x0))
    N = size(y, 2)
    yh = zeros(T, sys.ny, N)
    x = T.(copy(x0))
    @inbounds for t in 1:N
        φ = sys.basis(λ[t])
        At = _contract(sys.θ.A, φ)
        Bt = _contract(sys.θ.B, φ)
        Ct = _contract(sys.θ.C, φ)
        Dt = _contract(sys.θ.D, φ)
        ut = @view u[:, t]
        yt = @view y[:, t]
        ŷ = Ct * x + Dt * ut
        yh[:, t] = ŷ
        e = yt - ŷ
        x = At * x + Bt * ut + sys.K * e
    end
    yh
end

# ----------------------------------------------------------------------------
# Warm-start: sliding-window subspace ID + modal-form coefficient regression
# ----------------------------------------------------------------------------

"""
    lpv_warmstart(d,  λ,  nx; basis, window = nothing, stride = nothing) -> ComponentArray
    lpv_warmstart(ds, λs, nx; basis, window = nothing, stride = nothing) -> ComponentArray

Produce an initial guess `θ⁰` for [`lpv_pem`](@ref) by

1. sliding a window across the dataset and running [`subspaceid`](@ref) on each;
2. bringing every local model to a common modal form via `modal_form`;
3. regressing the entries of `A, B, C, D` against `basis(λ̄_i)` by ordinary least
   squares, where `λ̄_i` is the mean of `λ` in window `i`.

If a vector of datasets `ds` together with the corresponding vector of
scheduling trajectories `λs` is provided, the same procedure is applied to each
dataset independently and the resulting local models are pooled before the
final regression. All datasets must share the same sample time, number of
inputs, and number of outputs.

`basis` follows the same convention as in [`lpv_pem`](@ref) (a `Function` returning
a vector, or a `Vector` of functions). `window` defaults to `max(20nx, N÷10)` and
`stride` defaults to `window ÷ 4`.

The modal-form alignment is a heuristic — it works well when the dominant poles
move smoothly with `λ` and do not change order or coalesce, but can mis-align
entries when poles cross. In those cases pass a hand-crafted `p0` to `lpv_pem`
instead.

Returns a `ComponentArray` with fields `A, B, C, D`, each a 3-D array whose
trailing dimension is the basis index.

!!! warning "Experimental"
    LPV identification in this package is considered experimental and may
    change in the future without respecting semantic versioning.
"""
lpv_warmstart(d::AbstractIdData, λ::AbstractVector, nx::Int; kwargs...) =
    lpv_warmstart([d], [λ], nx; kwargs...)

function lpv_warmstart(ds::AbstractVector{<:AbstractIdData},
                       λs::AbstractVector,
                       nx::Int;
                       basis,
                       window::Union{Nothing,Int} = nothing,
                       stride::Union{Nothing,Int} = nothing)
    length(ds) == length(λs) || throw(ArgumentError("Number of datasets must equal number of λ trajectories"))
    isempty(ds) && throw(ArgumentError("Must provide at least one dataset"))
    allequal(d.Ts for d in ds) || throw(ArgumentError("All datasets must share the same sample time"))
    ny, nu = ds[1].ny, ds[1].nu
    all(d.ny == ny for d in ds) || throw(ArgumentError("All datasets must have the same number of outputs"))
    all(d.nu == nu for d in ds) || throw(ArgumentError("All datasets must have the same number of inputs"))
    basis_fn, nb = _normalize_basis(basis, first(λs[1]))

    locals = StateSpace[]
    λ_means = Float64[]
    for (d, λ) in zip(ds, λs)
        length(λ) == length(d) || throw(ArgumentError("each λ must have the same length as its dataset"))
        N = length(d)
        w = window === nothing ? max(20nx, N ÷ 10) : window
        w = min(w, N)
        st = stride === nothing ? max(1, w ÷ 4) : stride
        starts = 1:st:(N - w + 1)
        isempty(starts) && (starts = 1:1)
        for s in starts
            d_i = d[s:(s + w - 1)]
            λ_i = @view λ[s:(s + w - 1)]
            try
                sys_i = subspaceid(d_i, nx; verbose = false)
                sysm, _, _ = modal_form(sys_i.sys)
                push!(locals, sysm)
                push!(λ_means, mean(λ_i))
            catch err
                @warn "Local fit failed for window starting at $s; skipping" exception = (err, catch_backtrace())
            end
        end
    end

    M = length(locals)
    M >= 1 || error("lpv_warmstart: no window produced a successful local fit")
    if M < nb
        @warn "lpv_warmstart: fewer valid windows ($M) than basis functions ($nb); the regression is underdetermined"
    end

    Φ = zeros(M, nb)
    for i in 1:M
        Φ[i, :] = basis_fn(λ_means[i])
    end

    ComponentArray(
        A = _ols_field(Φ, locals, s -> s.A),
        B = _ols_field(Φ, locals, s -> s.B),
        C = _ols_field(Φ, locals, s -> s.C),
        D = _ols_field(Φ, locals, s -> s.D),
    )
end

function _ols_field(Φ, locals, get)
    M = length(locals)
    nb = size(Φ, 2)
    nr, nc = size(get(locals[1]))
    arr = zeros(nr, nc, nb)
    for r in 1:nr, c in 1:nc
        b = [get(locals[i])[r, c] for i in 1:M]
        arr[r, c, :] = Φ \ b
    end
    arr
end

# ----------------------------------------------------------------------------
# lpv_pem
# ----------------------------------------------------------------------------

function _lpv_dataset_predloss(p, x0_col, basis_fn, λ, y, u, metric,
                                ::Val{zeroD}, ::Val{predflag}) where {zeroD, predflag}
    N = size(y, 2)
    x = copy(x0_col)
    L = zero(eltype(p))
    @inbounds for t in 1:N
        φ = basis_fn(λ[t])
        At = _contract(p.A, φ)
        Bt = _contract(p.B, φ)
        Ct = _contract(p.C, φ)
        ut = @view u[:, t]
        yt = @view y[:, t]
        if zeroD
            ŷ = Ct * x
        else
            Dt = _contract(p.D, φ)
            ŷ = Ct * x + Dt * ut
        end
        e = yt - ŷ
        L += sum(metric, e)
        if predflag
            x = At * x + Bt * ut + p.K * e
        else
            x = At * x + Bt * ut
        end
    end
    L
end

# Single-dataset loss: x0 is a Vector (column) in the ComponentArray.
function _lpv_loss_factory(zeroD_v::Val, predflag_v::Val, basis_fn, λ, y, u, metric, regularizer)
    let basis_fn = basis_fn, λ = λ, y = y, u = u, metric = metric, regularizer = regularizer
        function loss(p)
            L = _lpv_dataset_predloss(p, p.x0, basis_fn, λ, y, u, metric, zeroD_v, predflag_v)
            L + regularizer(p)
        end
        return loss
    end
end

# Multi-dataset loss: x0 is a Matrix of shape (nx, M); column i is the initial state for dataset i.
function _lpv_multi_loss_factory(zeroD_v::Val, predflag_v::Val, basis_fn, λs, ys, us, metric, regularizer)
    let basis_fn = basis_fn, λs = λs, ys = ys, us = us, metric = metric, regularizer = regularizer
        function loss(p)
            L = zero(eltype(p))
            @inbounds for i in 1:length(λs)
                L += _lpv_dataset_predloss(p, view(p.x0, :, i), basis_fn,
                                            λs[i], ys[i], us[i], metric,
                                            zeroD_v, predflag_v)
            end
            L + regularizer(p)
        end
        return loss
    end
end

"""
    sys, x0, res = lpv_pem( d,  λ,  nx; basis, ...)
    sys, x0, res = lpv_pem(ds, λs, nx; basis, ...)

Linear Parameter-Varying (LPV) state-space identification using PEM.

The model has matrices that vary as a basis expansion in a measured scalar
scheduling variable `λ(t)`:

    A(λ) = Σ_k θ^A_k φ_k(λ),     ...,     D(λ) = Σ_k θ^D_k φ_k(λ).

A constant Kalman gain `K` is also estimated (when `focus = :prediction`).
Estimation minimizes one-step prediction error over the full dataset; the
predictor is time-varying along `λ(t)`.

If a vector of datasets `ds` and a corresponding vector of scheduling
trajectories `λs` is passed, the same parameters `θ` and `K` are fit to all
datasets jointly. Each dataset gets its own initial state, returned as a
matrix `x0::Matrix{Float64}` of shape `(nx, length(ds))`. All datasets must
share sample time, number of inputs, and number of outputs.

# Arguments
- `d`, `ds`: [`iddata`](@ref) (or a vector thereof).
- `λ::AbstractVector`, `λs`: scheduling trajectory (or a vector of them);
  each must satisfy `length(λ) == length(d)`.
- `nx`: model order.

# Keyword arguments
- `basis`: either a `Vector` of functions `λ -> ::Real` or a single `Function`
  `λ -> ::AbstractVector`. The number of basis functions is inferred.
- `focus`: `:prediction` (default) or `:simulation`. In `:simulation` mode `K` is
  forced to zero and the parameter `K` is dropped.
- `zeroD`: if `true`, the `D(λ)` term is omitted entirely.
- `p0`: optional initial guess as a `ComponentArray` with fields `A,B,C[,D]` of
  shape `(nx,nx,nb),(nx,nu,nb),(ny,nx,nb),(ny,nu,nb)`. If `nothing`, a warm
  start is computed via [`lpv_warmstart`](@ref).
- `K0`: optional initial Kalman gain `(nx,ny)`.
- `x0`: optional initial state. For the multi-dataset method, this is a
  matrix of shape `(nx, length(ds))` with one column per experiment.
- The remaining keyword arguments are forwarded to `Optim.Options` and follow
  the same conventions as [`structured_pem`](@ref) and [`newpem`](@ref).

# Returns
A named tuple `(; sys, x0, res)` where `sys::LPVStateSpace`, `x0` is the
optimized initial state (a `Vector` for the single-dataset method, a `Matrix`
for the multi-dataset method), and `res` is the `Optim` result.

# Example
```julia
A0 = [0.9 0.1; 0 0.8]; A1 = [0.0 0.0; 0.05 0.0]
B0 = [0.1; 0.2;;];     B1 = [0.0; 0.05;;]
C  = [1.0 0.0]; Ts = 0.05
T  = 2000
λ  = sin.(0.01 .* (1:T))
u  = randn(1, T)
A(λ) = A0 .+ λ .* A1
B(λ) = B0 .+ λ .* B1

x = zeros(2); y = zeros(1, T)
for t in 1:T
    y[:, t] = C * x
    x = A(λ[t]) * x + B(λ[t]) * u[:, t]
end
y .+= 0.01 .* randn(size(y))
d = iddata(y, u, Ts)

basis = [_ -> 1.0, λ -> λ]
sys, x0h, res = lpv_pem(d, λ, 2; basis)
```

See also [`lpv_warmstart`](@ref), [`structured_pem`](@ref), [`newpem`](@ref).

!!! warning "Experimental"
    LPV identification in this package is considered experimental and may
    change in the future without respecting semantic versioning. The basis
    API, the handling of the Kalman gain `K`, and the return type
    [`LPVStateSpace`](@ref) in particular are subject to revision; the
    `h > 1` prediction horizon, multi-dimensional scheduling variables, and
    λ-varying `K` are also not yet supported.
"""
# Single-dataset wrapper: delegates to the multi-dataset implementation and unwraps x0 back to a Vector.
function lpv_pem(d::AbstractIdData, λ::AbstractVector, nx::Int;
                 x0 = nothing, kwargs...)
    x0_mat = x0 === nothing ? nothing : reshape(collect(x0), :, 1)
    out = lpv_pem([d], [λ], nx; x0 = x0_mat, kwargs...)
    (; sys = out.sys, x0 = vec(out.x0), res = out.res)
end

function lpv_pem(ds::AbstractVector{<:AbstractIdData},
                 λs::AbstractVector,
                 nx::Int;
                 basis,
                 focus::Symbol = :prediction,
                 zeroD::Bool = false,
                 p0 = nothing,
                 K0 = nothing,
                 x0 = nothing,
                 h::Int = 1,
                 metric::F = abs2,
                 regularizer::RE = p -> 0,
                 optimizer = BFGS(linesearch = LineSearches.BackTracking()),
                 store_trace = true,
                 show_trace = true,
                 show_every = 50,
                 iterations = 10000,
                 allow_f_increases = false,
                 time_limit = 100,
                 x_tol = 0,
                 x_abstol = x_tol,
                 f_abstol = 1e-16,
                 g_tol = 1e-12,
                 f_calls_limit = 0,
                 g_calls_limit = 0,
                 ) where {F,RE}
    h == 1 || throw(ArgumentError("h > 1 not supported for lpv_pem yet"))
    focus ∈ (:prediction, :simulation) || throw(ArgumentError("focus must be :prediction or :simulation"))
    length(ds) == length(λs) || throw(ArgumentError("Number of datasets must equal number of λ trajectories"))
    isempty(ds) && throw(ArgumentError("Must provide at least one dataset"))
    allequal(d.Ts for d in ds) || throw(ArgumentError("All datasets must share the same sample time"))
    ny, nu = ds[1].ny, ds[1].nu
    all(d.ny == ny for d in ds) || throw(ArgumentError("All datasets must have the same number of outputs"))
    all(d.nu == nu for d in ds) || throw(ArgumentError("All datasets must have the same number of inputs"))
    for i in 1:length(ds)
        length(λs[i]) == length(ds[i]) || throw(ArgumentError("length(λs[$i]) must equal length(ds[$i])"))
    end

    M = length(ds)
    Ts = ds[1].Ts
    basis_fn, nb = _normalize_basis(basis, first(λs[1]))

    if p0 === nothing
        show_trace && @info "lpv_pem: computing warm-start with sliding-window subspaceid"
        p0 = lpv_warmstart(ds, λs, nx; basis = basis_fn)
    end

    K0_ = K0 === nothing ?
        (focus === :prediction ? 1e-6 .* randn(nx, ny) : zeros(nx, ny)) :
        copy(K0)
    size(K0_) == (nx, ny) || throw(DimensionMismatch("K0 must have size ($nx, $ny)"))

    x0_mat = if x0 === nothing
        m = zeros(nx, M)
        for (i, (d, λ)) in enumerate(zip(ds, λs))
            sys_mean = let
                φ = basis_fn(mean(λ))
                A_mean = _contract(p0.A, φ)
                B_mean = _contract(p0.B, φ)
                C_mean = _contract(p0.C, φ)
                D_mean = zeroD ? zeros(ny, nu) : _contract(p0.D, φ)
                ss(A_mean, B_mean, C_mean, D_mean, Ts)
            end
            m[:, i] = try
                estimate_x0(sys_mean, d, min(length(d), 10nx))
            catch
                zeros(nx)
            end
        end
        m
    else
        size(x0) == (nx, M) || throw(DimensionMismatch("x0 must have size ($nx, $M)"))
        collect(x0)
    end

    if zeroD
        p_init = ComponentArray(
            A = collect(p0.A),
            B = collect(p0.B),
            C = collect(p0.C),
            K = K0_,
            x0 = x0_mat,
        )
    else
        D0 = hasproperty(p0, :D) ? collect(p0.D) : zeros(ny, nu, nb)
        p_init = ComponentArray(
            A = collect(p0.A),
            B = collect(p0.B),
            C = collect(p0.C),
            D = D0,
            K = K0_,
            x0 = x0_mat,
        )
    end

    ys = [time2(output(d)) for d in ds]
    us = [time2(input(d)) for d in ds]

    loss = _lpv_multi_loss_factory(
        Val(zeroD),
        Val(focus === :prediction),
        basis_fn, λs, ys, us, metric, regularizer,
    )

    res = Optim.optimize(
        loss,
        p_init,
        optimizer,
        Optim.Options(;
            store_trace, show_trace, show_every, iterations, allow_f_increases,
            time_limit, x_abstol, f_abstol, g_tol, f_calls_limit, g_calls_limit);
        autodiff = AutoForwardDiff(),
    )
    p_opt = res.minimizer

    θ_opt = if zeroD
        ComponentArray(
            A = collect(p_opt.A),
            B = collect(p_opt.B),
            C = collect(p_opt.C),
            D = zeros(ny, nu, nb),
        )
    else
        ComponentArray(
            A = collect(p_opt.A),
            B = collect(p_opt.B),
            C = collect(p_opt.C),
            D = collect(p_opt.D),
        )
    end

    K_opt = focus === :prediction ? collect(p_opt.K) : zeros(nx, ny)
    sys = LPVStateSpace(basis_fn, nb, θ_opt, K_opt, Ts, nx, nu, ny)
    (; sys, x0 = collect(p_opt.x0), res)
end
