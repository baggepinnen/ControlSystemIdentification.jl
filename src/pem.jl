"""
This function is deprecated, see [`newpem`](@ref)
"""
function pem(
    d;
    nx,
    solver = BFGS(),
    difficult = false,
    A = 0.0001randn(nx, nx),
    B = 0.001randn(nx, obslength(input(d))),
    K = 0.001randn(nx, obslength(output(d))),
    x0 = nothing,
    kwargs...,
)

    if any(!isnothing, (A,B,K))
        A === nothing ? 0.0001randn(nx, nx) : A
        B === nothing ? 0.001randn(nx, obslength(input(d))) : B
        K === nothing ? 0.001randn(nx, obslength(output(d))) : K
        C = [I zeros(d.ny, nx - d.ny)]
        sys0 = PredictionStateSpace(ss(A, B, C, 0, d.Ts), K)
        return newpem(d, nx;
            optimizer = solver,
            sys0,
            kwargs...
        )
    end
    Base.depwarn("This version of pem is deprecated, use newpem instead", :pem)
    newpem(d, nx;
        optimizer = solver,
        kwargs...
    )
end



using Optim, Optim.LineSearches

"""
    sys, x0, res = newpem(
        d,
        nx;
        zeroD  = true,
        focus  = :prediction,
        stable = true,
        sys0   = subspaceid(d, nx; zeroD, focus, stable),
        metric = abs2,
        regularizer = (p, P) -> 0,
        optimizer = BFGS(
            linesearch = LineSearches.BackTracking(),
        ),
        store_trace = true,
        show_trace  = true,
        show_every  = 50,
        iterations  = 10000,
        time_limit  = 100,
        x_tol       = 0,
        f_abstol    = 0,
        g_tol       = 1e-12,
        f_calls_limit = 0,
        g_calls_limit = 0,
        allow_f_increases = false,
    )

A new implementation of the prediction-error method (PEM). Note that this is an experimental implementation and subject to breaking changes not respecting semver.

The prediction-error method is an iterative, gradient-based optimization problem, as such, it can be extra sensitive to signal scaling, and it's recommended to perform scaling to `d` before estimation, e.g., by pre and post-multiplying with diagonal matrices `d̃ = Dy*d*Du`, and apply the inverse scaling to the resulting system. In this case, we have
```math
D_y y = G̃ D_u u ↔ y = D_y^{-1} G̃ D_u u
```
hence `G = Dy \\ G̃ * Du` where \$ G̃ \$ is the plant estimated for the scaled iddata.

# Arguments:
- `d`: [`iddata`](@ref)
- `nx`: Model order
- `zeroD`: Force zero `D` matrix
- `stable` if true, stability of the estimated system will be enforced by eigenvalue reflection using [`schur_stab`](@ref) with `ϵ=Ts/100` (default). If `stable` is a real value, the value is used instead of the default `ϵ`.
- `sys0`: Initial guess, if non provided, [`subspaceid`](@ref) is used as initial guess.
- `focus`: `prediction` or `:simulation`. If `:simulation`, hte `K` matrix will be zero.
- `optimizer`: One of Optim's optimizers
- `metric`: The metric used to measure residuals. Try, e.g., `abs` for better resistance to outliers.
The rest of the arguments are related to `Optim.Options`.
- `regularizer`: A function of the parameter vector and the corresponding `PredictionStateSpace/StateSpace` system that can be used to regularize the estimate.

# Example
```
using ControlSystemIdentification, ControlSystems, Plots
G = DemoSystems.doylesat()
T = 1000  # Number of time steps
Ts = 0.01 # Sample time
sys = c2d(G, Ts)
nx = sys.nx
nu = sys.nu
ny = sys.ny
x0 = zeros(nx) # actual initial state
sim(sys, u, x0 = x0) = lsim(sys, u; x0)[1]

σy = 1e-1 # Noise covariance

u  = randn(nu, T)
y  = sim(sys, u, x0)
yn = y .+ σy .* randn.() # Add measurement noise
d  = iddata(yn, u, Ts)

sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, show_every=10)

plot(
    bodeplot([sys, sysh]),
    predplot(sysh, d, x0h), # Include the estimated initial state in the prediction
)
```

# Extended help
This implementation uses a tridiagonal parametrization of the A-matrix that has been shown to be favourable from an optimization perspective.¹ The initial guess `sys0` is automatically transformed to a special tridiagonal modal form. 
[1]: Mckelvey, Tomas & Helmersson, Anders. (1997). State-space parametrizations of multivariable linear systems using tridiagonal matrix forms.

The parameter vector used in the optimizaiton takes the following form
```julia
p = [trivec(A); vec(B); vec(C); vec(D); vec(K); vec(x0)]
```
Where `ControlSystemIdentification.trivec` vectorizes the `-1,0,1` diagonals of `A`. If `focus = :simulation`, `K` is omitted, and if `zeroD = true`, `D` is omitted.
"""
function newpem(
    d,
    nx;
    zeroD = true,
    focus = :prediction,
    h = 1,
    stable = true,
    sys0 = subspaceid(d, nx; zeroD, focus, stable),
    metric::F = abs2,
    regularizer::RE = (p, P) -> 0,
    optimizer = BFGS(
        # alphaguess = LineSearches.InitialStatic(alpha = 0.95),
        linesearch = LineSearches.BackTracking(),
    ),
    store_trace = true,
    show_trace = true,
    show_every = 50,
    iterations = 10000,
    allow_f_increases = false,
    time_limit = 100,
    x_tol = 0,
    f_abstol = 1e-16,
    g_tol = 1e-12,
    f_calls_limit = 0,
    g_calls_limit = 0,
    safe = false,
) where {F, RE}
    T = promote_type(eltype(d.y), eltype(sys0.A))
    nu = d.nu
    ny = d.ny
    sys0, Tmodal = modal_form(sys0)
    A, B, C, D = ssdata(sys0)
    K = if hasfield(typeof(sys0), :K)
        convert(Matrix{T}, sys0.K)
    else
        zeros(T, nx, ny)
    end
    pred = focus === :prediction
    pd = predictiondata(d)
    x0i::Vector{T} = if h == 1 || focus === :simulation
        T.(estimate_x0(sys0, d, min(length(d), 10nx)))
    else
        T.(estimate_x0(prediction_error(PredictionStateSpace(sys0, K, 0, 0); h), pd, min(length(pd), 10nx)))
    end
    p0::Vector{T} = if pred
        T.(zeroD ? [trivec(A); vec(B); vec(C); vec(K); vec(x0i)] : [trivec(A); vec(B); vec(C); vec(D); vec(K); vec(x0i)])
    else
        T.(zeroD ? [trivec(A); vec(B); vec(C); vec(x0i)] : [trivec(A); vec(B); vec(C); vec(D); vec(x0i)])
    end
    D0 = zeros(T, ny, nu)
    function predloss(p)
        sysi, Ki, x0 = vec2modal(p, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K)
        syso = PredictionStateSpace(sysi, Ki, 0, 0)
        Pe = prediction_error(syso; h)
        e, _ = lsim(Pe, pd; x0)
        unstab = maximum(abs, eigvals(ForwardDiff.value.(syso.A - syso.K*syso.C))) >= 1
        c1 = sum(metric, e)
        c1 + min(10*ForwardDiff.value(c1)*unstab, 1e6) + regularizer(p, syso)
    end
    function simloss(p)
        syssim, _, x0 = vec2modal(p, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K)
        y, _ = lsim(syssim, d; x0)
        y .= metric.(y .- d.y)
        sum(y) + regularizer(p, syssim)
    end
    local res, sys_opt, K_opt, x0_opt
    try
        res = Optim.optimize(
            pred ? predloss : simloss,
            p0,
            optimizer,
            Optim.Options(;
                store_trace, show_trace, show_every, iterations, allow_f_increases,
                time_limit, x_tol, f_abstol, g_tol, f_calls_limit, g_calls_limit),
            autodiff = :forward,
        )
        sys_opt::StateSpace{Discrete{T}, T}, K_opt::Matrix{T}, x0_opt = vec2modal(res.minimizer, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K)
    catch err
        if safe
            @error "Optimization failed, returning initial estimate." err
        else
            @error "Optimization failed, call `newpem(...; safe=true) to exit gracefully returning the initial estimate."
            rethrow()
        end
        sys_opt = sys0
        K_opt = sys0.K
        x0_opt = x0i
        res = nothing
    end
    if stable > 0 && !isstable(sys_opt)
        @warn("Estimated system dynamics A is unstable, stabilizing A using eigenvalue reflection")
        Astab = schur_stab(sys_opt.A, stable === true ? sys0.Ts/100 : stable)
        sys_opt.A .= Astab
    end
    sysp_opt = PredictionStateSpace(
        sys_opt,
        pred ? K_opt : zeros(T, nx, ny),
        zeros(T, nx, nx),
        zeros(T, ny, ny),
        zeros(T, nx, ny),
    )
    pred && !isstable(observer_predictor(sysp_opt)) && @warn("Estimated predictor dynamics A-KC is unstable")
    e2, _ = pred ? lsim(prediction_error(sysp_opt), pd) : lsim(sys_opt, d)
    R = cov(e2, dims = 2)
    mul!(sysp_opt.S, K_opt, R)
    Q0 = sysp_opt.S * K_opt'
    Q = Hermitian(Q0 + eps(maximum(abs, Q0)) * I)
    # Note: to get the correct Q so that kalman(sys, Q, R) = K, one must solve an LMI problem to minimize the cross term S. If one tolerates S, we have kalman(sys, Q, R, S) = K, where S = K*R. Without S, Q can be *very* far from correct.
    sysp_opt.R .= R
    sysp_opt.Q .= Q
    sysp_opt, x0_opt, res
end

# this method fails for duals so this is an overload to silence the warning and save some time
ControlSystems.balance_statespace(A::AbstractMatrix{<:ForwardDiff.Dual}, B::AbstractMatrix, C::AbstractMatrix, perm::Bool=false) = A,B,C,I

trivec(A) = [A[diagind(A, -1)]; A[diagind(A, 0)]; A[diagind(A, 1)]]

function vec2modal(p, ny, nu, nx, timeevol, zeroD, pred, D0, K0)
    if nx > 1
        al = (1:nx-1)
        a  = (1:nx) .+ al[end]
        au = (1:nx-1) .+ a[end]
        A = Matrix(Tridiagonal(p[al], p[a], p[au]))
    else
        A = [p[1];;]
        au = 1:1
    end
    bi = (1:nx*nu) .+ au[end]
    ci = (1:nx*ny) .+ bi[end]
    B = reshape(p[bi], nx, nu)
    C = reshape(p[ci], ny, nx)
    if zeroD
        di = (0:0) .+ ci[end]
        D = D0
    else
        di = (1:nu*ny) .+ ci[end]
        D = reshape(p[di], ny, nu)
    end
    if pred
        ki = (1:nx*ny) .+ di[end]
        K = reshape(p[ki], nx, ny)
    else
        ki = (0:0) .+ di[end]
        K = K0
    end
    x0 = p[ki[end]+1:end] # x0 may belong to an extended predictor state and can have any length
    ss(A, B, C, D, timeevol), K, x0
end

function modal2vec(sys)
    [
        sys.A[diagind(sys.A, -1)]
        sys.A[diagind(sys.A, 0)]
        sys.A[diagind(sys.A, 1)]
        vec(sys.B)
        vec(sys.C)
        vec(sys.D)
    ]
end

# function vec2sys(v::AbstractArray, ny::Int, nu::Int, ts=nothing)
#     n = length(v)
#     p = (ny+nu)
#     nx = Int(-p/2 + sqrt(p^2 - 4nu*ny + 4n)/2)
#     @assert n == nx^2 + nx*nu + ny*nx + ny*nu
#     ai = (1:nx^2)
#     bi = (1:nx*nu) .+ ai[end]
#     ci = (1:nx*ny) .+ bi[end]
#     di = (1:nu*ny) .+ ci[end]
#     A = reshape(v[ai], nx, nx)
#     B = reshape(v[bi], nx, nu)
#     C = reshape(v[ci], ny, nx)
#     D = reshape(v[di], ny, nu)
#     ts === nothing ? ss(A, B, C, D) : ss(A, B, C, D, ts)
# end

# function Base.vec(sys::LTISystem)
#     [vec(sys.A); vec(sys.B); vec(sys.C); vec(sys.D)]
# end


# """
#     ssest(data::FRD, p0; opt = BFGS(), modal = false, opts = Optim.Options(store_trace = true, show_trace = true, show_every = 5, iterations = 2000, allow_f_increases = false, time_limit = 100, x_tol = 1.0e-5, f_tol = 0, g_tol = 1.0e-8, f_calls_limit = 0, g_calls_limit = 0))

# Estimate a statespace model from frequency-domain data.

# It's often a good idea to start with `opt = NelderMead()` and then refine with `opt = BFGS()`.

# # Arguments:
# - `data`: DESCRIPTION
# - `p0`: Statespace model of Initial guess
# - `opt`: Optimizer
# - `modal`: indicate whether or not to estimate the model with a tridiagonal A matrix. This reduces the number of parameters (for nx >= 3).
# - `opts`: `Optim.Options`
# """
# function ssest(data::FRD, p0; 
#     # freq_weight = 1 ./ (data.w .+ data.w[2]),
#     opt = BFGS(),
#     modal = false,
#     opts = Optim.Options(
#         store_trace       = true,
#         show_trace        = true,
#         show_every        = 5,
#         iterations        = 2000,
#         allow_f_increases = false,
#         time_limit        = 100,
#         x_tol             = 1e-5,
#         f_tol             = 0,
#         g_tol             = 1e-8,
#         f_calls_limit     = 0,
#         g_calls_limit     = 0,
#     ),
# )


#     function loss(p)
#         if modal
#             sys = vec2modal(p, p0.ny, p0.nu, p0.nx, p0.timeevol)
#         else
#             sys = vec2sys(p, p0.ny, p0.nu)
#         end
#         F = freqresp(sys, data.w).parent
#         F .-= data.r
#         mean(abs2, F)
#     end
    
#     res = Optim.optimize(
#         loss,
#         modal ? modal2vec(p0) : vec(p0),
#         opt,
#         opts,
#         autodiff=:forward
#     )
#     (modal ? vec2modal(res.minimizer, p0.ny, p0.nu, p0.nx, p0.timeevol) : 
#         vec2sys(res.minimizer, p0.ny, p0.nu)), res
# end




# using ControlSystems, ControlSystemIdentification, Optim
# G = modal_form(ssrand(2,3,4))[1]
# w = exp10.(LinRange(-2, 2, 30))
# r = freqresp(G, w).parent
# data = FRD(w, r)


# G0 = modal_form(ssrand(2,3,4))[1]
# Gh, _ = ControlSystemIdentification.ssest(data, G0, opt=NelderMead(), modal=true)
# Gh, _ = ControlSystemIdentification.ssest(data, Gh, modal=true)


# bodeplot(G, w)
# bodeplot!(Gh, w)






## modal_form from RobustAndOptimalControl, to be removed when ROC is faster to load and https://github.com/andreasvarga/DescriptorSystems.jl/issues/8 is resolved

"""
    Db,Vb,E = blockdiagonalize(A::AbstractMatrix)

`Db` is a block-diagonal matrix and `Vb` is the corresponding "eigenvectors" such that `Vb*Db/Vb = A`
"""
function blockdiagonalize(A::AbstractMatrix)
    E = eigen(A, sortby=eigsortby)
    Db,Vb = cdf2rdf(E)
    Db,Vb,E
end

eigsortby(λ::Real) = λ
eigsortby(λ::Complex) = (abs(imag(λ)),real(λ))

function complex_indices(A::Matrix) # assumes A on block diagonal form
    findall(diag(A, -1) .!= 0)
end

function real_indices(A::Matrix) # assumes A on block diagonal form
    size(A,1) == 1 && return [1]
    setdiff(findall(diag(A, -1) .== 0), complex_indices(A).+1)
end

function complex_indices(D::AbstractVector)
    complex_eigs = imag.(D) .!= 0
    findall(complex_eigs)
end

function cdf2rdf(E::Eigen)
    # Implementation inspired by scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cdf2rdf.html
    # with the licence https://github.com/scipy/scipy/blob/v1.6.3/LICENSE.txt
    D,V = E
    n = length(D)

    # get indices for each first pair of complex eigenvalues
    complex_inds = complex_indices(D)

    # eigvals are sorted so conjugate pairs are next to each other
    j = complex_inds[1:2:end]
    k = complex_inds[2:2:end]

    # put real parts on diagonal
    Db = zeros(n, n)
    Db[diagind(Db)] .= real(D)

    # compute eigenvectors for real block diagonal eigenvalues
    U = zeros(eltype(D), n, n)
    U[diagind(U)] .= 1.0

    # transform complex eigvals to real blockdiag form
    for (k,j) in zip(k,j)
        Db[j, k] = imag(D[j]) # put imaginary parts in blocks 
        Db[k, j] = imag(D[k])

        U[j, j] = 0.5im
        U[j, k] = 0.5
        U[k, j] = -0.5im
        U[k, k] = 0.5
    end
    Vb = real(V*U)

    return Db, Vb
end

"""
    sysm, T, E = modal_form(sys; C1 = false)

Bring `sys` to modal form.

The modal form is characterized by being tridiagonal with the real values of eigenvalues of `A` on the main diagonal and the complex parts on the first sub and super diagonals. `T` is the similarity transform applied to the system such that 
```julia
sysm ≈ similarity_transform(sys, T)
```

If `C1`, then an additional convention for SISO systems is used, that the `C`-matrix coefficient of real eigenvalues is 1. If `C1 = false`, the `B` and `C` coefficients are chosen in a balanced fashion.

`E` is an eigen factorization of `A`.

See also [`hess_form`](@ref) and [`schur_form`](@ref)
"""
function modal_form(sys; C1 = false)
    Ab,T,E = blockdiagonalize(sys.A)
    Ty = eltype(Ab)
    # Calling similarity_transform looks like a detour, but this implementation allows modal_form to work with any AbstractStateSpace which implements a custom method for similarity transform
    sysm = similarity_transform(sys, T)
    sysm.A .= Ab # sysm.A should already be Ab after similarity_transform, but Ab has less numerical noise
    if ControlSystems.issiso(sysm)
        # This enforces a convention: the C matrix entry for the first component in each mode is positive. This allows SISO systems on modal form to be interpolated in a meaningful way by interpolating their coefficients. 
        # Ref: "New Metrics Between Rational Spectra and their Connection to Optimal Transport" , Bagge Carlson,  Chitre
        ci = complex_indices(sysm.A)
        flips = ones(Ty, sysm.nx)
        for i in ci
            if sysm.C[1, i] < 0
                flips[i] = -1
                flips[i .+ 1] = -1
            end
        end
        ri = real_indices(sysm.A)
        for i in ri
            c = sysm.C[1, i]
            if C1
                if c != 0
                    flips[i] /= c
                end
            else
                b = sysm.B[i, 1]
                flips[i] *= sqrt(abs(b))/(sqrt(abs(c)) + eps(b))
            end
        end
        T2 = diagm(flips)
        sysm = similarity_transform(sysm, T2)
        T = T*T2
        sysm.A .= Ab # Ab unchanged by diagonal T
    end
    sysm, T, E
end