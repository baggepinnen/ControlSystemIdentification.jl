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
    lpv_warmstart(d, λ, nx; basis, window = nothing, stride = nothing) -> ComponentArray

Produce an initial guess `θ⁰` for [`lpv_pem`](@ref) by

1. sliding a window across the dataset and running [`subspaceid`](@ref) on each;
2. bringing every local model to a common modal form via `modal_form`;
3. regressing the entries of `A, B, C, D` against `basis(λ̄_i)` by ordinary least
   squares, where `λ̄_i` is the mean of `λ` in window `i`.

`basis` follows the same convention as in [`lpv_pem`](@ref) (a `Function` returning
a vector, or a `Vector` of functions). `window` defaults to `max(20nx, N÷10)` and
`stride` defaults to `window ÷ 4`.

The modal-form alignment is a heuristic — it works well when the dominant poles
move smoothly with `λ` and do not change order or coalesce, but can mis-align
entries when poles cross. In those cases pass a hand-crafted `p0` to `lpv_pem`
instead.

Returns a `ComponentArray` with fields `A, B, C, D`, each a 3-D array whose
trailing dimension is the basis index.
"""
function lpv_warmstart(d::AbstractIdData, λ::AbstractVector, nx::Int;
                       basis,
                       window::Union{Nothing,Int} = nothing,
                       stride::Union{Nothing,Int} = nothing)
    length(λ) == length(d) || throw(ArgumentError("length(λ) must equal length(d)"))
    basis_fn, nb = _normalize_basis(basis, first(λ))
    N = length(d)
    ny, nu = d.ny, d.nu

    window === nothing && (window = max(20nx, N ÷ 10))
    window = min(window, N)
    stride === nothing && (stride = max(1, window ÷ 4))

    starts = 1:stride:(N - window + 1)
    isempty(starts) && (starts = 1:1)

    locals = StateSpace[]
    λ_means = Float64[]
    for w in starts
        d_i = d[w:(w + window - 1)]
        λ_i = @view λ[w:(w + window - 1)]
        try
            sys_i = subspaceid(d_i, nx; verbose = false)
            sysm, _, _ = modal_form(sys_i.sys)
            push!(locals, sysm)
            push!(λ_means, mean(λ_i))
        catch err
            @warn "Local fit failed for window starting at $w; skipping" exception = (err, catch_backtrace())
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

function _lpv_loss_factory(::Val{zeroD}, ::Val{predflag}, basis_fn, λ, y, u, metric, regularizer) where {zeroD, predflag}
    let basis_fn = basis_fn, λ = λ, y = y, u = u, metric = metric, regularizer = regularizer
        function loss(p)
            N = size(y, 2)
            x = copy(p.x0)
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
            L + regularizer(p)
        end
        return loss
    end
end

"""
    sys, x0, res = lpv_pem(
        d, λ, nx;
        basis,
        focus = :prediction,
        zeroD = false,
        p0 = nothing,
        K0 = nothing,
        x0 = nothing,
        h = 1,
        metric = abs2,
        regularizer = p -> 0,
        optimizer = BFGS(linesearch = LineSearches.BackTracking()),
        store_trace = true, show_trace = true, show_every = 50,
        iterations = 10000, allow_f_increases = false,
        time_limit = 100, x_tol = 0, f_abstol = 1e-16, g_tol = 1e-12,
        f_calls_limit = 0, g_calls_limit = 0,
    )

Linear Parameter-Varying (LPV) state-space identification using PEM.

The model has matrices that vary as a basis expansion in a measured scalar
scheduling variable `λ(t)`:

    A(λ) = Σ_k θ^A_k φ_k(λ),     ...,     D(λ) = Σ_k θ^D_k φ_k(λ).

A constant Kalman gain `K` is also estimated (when `focus = :prediction`).
Estimation minimizes one-step prediction error over the full dataset; the
predictor is time-varying along `λ(t)`.

# Arguments
- `d`: [`iddata`](@ref).
- `λ::AbstractVector`: scheduling trajectory, `length(λ) == length(d)`.
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
- `x0`: optional initial state.
- The remaining keyword arguments are forwarded to `Optim.Options` and follow
  the same conventions as [`structured_pem`](@ref) and [`newpem`](@ref).

# Returns
A named tuple `(; sys, x0, res)` where `sys::LPVStateSpace`, `x0` is the
optimized initial state, and `res` is the `Optim` result.

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
"""
function lpv_pem(d::AbstractIdData, λ::AbstractVector, nx::Int;
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
    length(λ) == length(d) || throw(ArgumentError("length(λ) must equal length(d)"))

    basis_fn, nb = _normalize_basis(basis, first(λ))
    ny, nu = d.ny, d.nu

    if p0 === nothing
        show_trace && @info "lpv_pem: computing warm-start with sliding-window subspaceid"
        p0 = lpv_warmstart(d, λ, nx; basis = basis_fn)
    end

    K0_ = K0 === nothing ?
        (focus === :prediction ? 1e-6 .* randn(nx, ny) : zeros(nx, ny)) :
        copy(K0)
    size(K0_) == (nx, ny) || throw(DimensionMismatch("K0 must have size ($nx, $ny)"))

    x0_init = if x0 === nothing
        mean_λ = mean(λ)
        sys_mean = let
            φ = basis_fn(mean_λ)
            A_mean = _contract(p0.A, φ)
            B_mean = _contract(p0.B, φ)
            C_mean = _contract(p0.C, φ)
            D_mean = zeroD ? zeros(ny, nu) : _contract(p0.D, φ)
            ss(A_mean, B_mean, C_mean, D_mean, d.Ts)
        end
        try
            estimate_x0(sys_mean, d, min(length(d), 10nx))
        catch
            zeros(nx)
        end
    else
        length(x0) == nx || throw(DimensionMismatch("x0 must have length $nx"))
        copy(x0)
    end

    if zeroD
        p_init = ComponentArray(
            A = collect(p0.A),
            B = collect(p0.B),
            C = collect(p0.C),
            K = K0_,
            x0 = collect(x0_init),
        )
    else
        D0 = hasproperty(p0, :D) ? collect(p0.D) : zeros(ny, nu, nb)
        p_init = ComponentArray(
            A = collect(p0.A),
            B = collect(p0.B),
            C = collect(p0.C),
            D = D0,
            K = K0_,
            x0 = collect(x0_init),
        )
    end

    y = time2(output(d))
    u = time2(input(d))

    loss = _lpv_loss_factory(
        Val(zeroD),
        Val(focus === :prediction),
        basis_fn, λ, y, u, metric, regularizer,
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
    sys = LPVStateSpace(basis_fn, nb, θ_opt, K_opt, d.Ts, nx, nu, ny)
    (; sys, x0 = collect(p_opt.x0), res)
end
