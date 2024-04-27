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
        output_nonlinearity = nothing,
        input_nonlinearity = nothing,
        nlp = nothing,
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
hence `G = Dy \\ G̃ * Du` where \$ G̃ \$ is the plant estimated for the scaled iddata. Example:
```julia
Dy = Diagonal(1 ./ vec(std(d.y, dims=2))) # Normalize variance
Du = Diagonal(1 ./ vec(std(d.u, dims=2))) # Normalize variance
d̃ = Dy * d * Du
```
If a manually provided initial guess `sys0`, this must also be scaled appropriately.

# Arguments:
- `d`: [`iddata`](@ref)
- `nx`: Model order
- `zeroD`: Force zero `D` matrix
- `stable` if true, stability of the estimated system will be enforced by eigenvalue reflection using [`schur_stab`](@ref) with `ϵ=1/100` (default). If `stable` is a real value, the value is used instead of the default `ϵ`.
- `sys0`: Initial guess, if non provided, [`subspaceid`](@ref) is used as initial guess.
- `focus`: `prediction` or `:simulation`. If `:simulation`, the `K` matrix will be zero.
- `optimizer`: One of Optim's optimizers
- `metric`: The metric used to measure residuals. Try, e.g., `abs` for better resistance to outliers.
The rest of the arguments are related to `Optim.Options`.
- `regularizer`: A function of the parameter vector and the corresponding `PredictionStateSpace/StateSpace` system that can be used to regularize the estimate.
- `output_nonlinearity`: A function of `(y::Vector, p)` that operates on the output signal at a single time point, `yₜ`, and modifies it in-place. See below for details. `p` is a vector of estimated parameters that can be optimized.
- `input_nonlinearity`: A function of `(u::Matrix, p)` that operates on the _entire_ input signal `u` at once and modifies it in-place. See below for details. `p` is a vector of estimated parameters that is shared with `output_nonlinearity`.
- `nlp`: Initial guess vector for nonlinear parameters. If `output_nonlinearity` is provided, this can optionally be provided.

# Nonlinear estimation
Nonlinear systems on Hammerstein-Wiener form, i.e., systems with a static input nonlinearity and a static output nonlinearity with a linear system inbetween, can be estimated as long as the nonlinearities are known. The procedure is
1. If there is **a known input nonlinearity**, manually apply the input nonlinearity to the input signal `u` _before_ estimation, i.e., use the nonlinearly transformed input in the [`iddata`](@ref) object `d`. If **the input nonlinearity has unknown parameters**, provide the input nonlinearity as a function using the keyword argument `input_nonlinearity` to `newpem`. This function is expected to operate on the entire (matrix) input signal `u` and modify it _in-place_.
2. If the output nonlinearity _is invertible_, apply the inverse to the output signal `y` _before_ estimation similar to above.
3. If the output nonlinearity _is not invertible_, provide the nonlinear output transformation as a function using the keyword argument `output_nonlinearity` to `newpem`. This function is expected to operate on the (vector) output signal `y` and modify it _in-place_. Example:
```julia
function output_nonlinearity(y, p)
    y[1] = y[1] + p[1]*y[1]^2       # Note how the incoming vector is modified in-place
    y[2] = abs(y[2])
end
```
Please note, `y = f(y)` does not change `y` in-place, but creates a new vector `y` and assigns it to the variable `y`. This is not what we want here.

The second argument to `input_nonlinearity` and `output_nonlinearity` is an (optional) vector of parameters that can be optimized. To use this option, pass the keyword argument `nlp` to `newpem` with a vector of initial guesses for the nonlinear parameters. The nonlinear parameters are shared between output and input nonlinearities, i.e., these two functions will receive the same vector of parameters.

The result of this estimation is the linear system _without_ the nonlinearities.

# Example
The following simulates data from a linear system and estimates a model. For an example of nonlinear identification, see the documentation.
```
using ControlSystemIdentification, ControlSystemsBase Plots
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

The returned model is of type `PredictionStateSpace` and contains the field `sys` with the system model, as well as covariance matrices and estimated Kalman gain for a Kalman filter.

See also [`structured_pem`](@ref) and [`nonlinear_pem`](@ref).

# Extended help
This implementation uses a tridiagonal parametrization of the A-matrix that has been shown to be favourable from an optimization perspective.¹ The initial guess `sys0` is automatically transformed to a special tridiagonal modal form. 
[1]: Mckelvey, Tomas & Helmersson, Anders. (1997). State-space parametrizations of multivariable linear systems using tridiagonal matrix forms.

The parameter vector used in the optimization takes the following form
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
    output_nonlinearity = nothing,
    input_nonlinearity = nothing,
    nlp = nothing,
    sys0 = output_nonlinearity === nothing ? subspaceid(d, nx; zeroD, focus, stable) : 0.001*ssrand(d.ny, d.nu, nx, proper=zeroD, Ts=d.Ts),
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
    nnl = nlp === nothing ? 0 : length(nlp)
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
        T.(estimate_x0(prediction_error_filter(PredictionStateSpace(sys0, K, 0, 0); h), pd, min(length(pd), 10nx)))
    end
    p0::Vector{T} = if pred
        T.(zeroD ? [trivec(A); vec(B); vec(C); vec(K); vec(x0i)] : [trivec(A); vec(B); vec(C); vec(D); vec(K); vec(x0i)])
    else
        T.(zeroD ? [trivec(A); vec(B); vec(C); vec(x0i)] : [trivec(A); vec(B); vec(C); vec(D); vec(x0i)])
    end
    if output_nonlinearity !== nothing && nlp !== nothing
       p0 = [p0; nlp]
    end
    D0 = zeros(T, ny, nu)
    function predloss(p)
        sysi, Ki, x0, nlpi = vec2modal(p, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K, nnl)
        pdi = if input_nonlinearity === nothing
            pd
        else
            predictiondata(iddata(d.y, input_nonlinearity(copy(d.u), nlpi), d.Ts))
        end
        syso = PredictionStateSpace(sysi, Ki, 0, 0)
        Pyh = ControlSystemsBase.observer_predictor(syso; h)
        yh, _ = lsim(Pyh, pdi; x0)
        unstab = maximum(abs, eigvals(ForwardDiff.value.(syso.A - syso.K*syso.C))) >= 1
        @views if output_nonlinearity !== nothing
            for i = axes(yh, 2)
                # NOTE: the output nonlinearity is applied after the prediction error correction which is likely suboptimal
                output_nonlinearity(yh[:, i], nlpi)
            end
        end
        yh .= metric.(yh .- d.y)
        c1 = sum(yh)
        c1 + min(10*ForwardDiff.value(c1)*unstab, 1e6) + regularizer(p, syso)
    end
    function simloss(p)
        syssim, _, x0, nlpi = vec2modal(p, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K, nnl)
        di = if input_nonlinearity === nothing
            d
        else
            iddata(d.y, input_nonlinearity(copy(d.u), nlpi), d.Ts)
        end
        y, _ = lsim(syssim, di; x0)
        @views if output_nonlinearity !== nothing
            for i = axes(y, 2)
                output_nonlinearity(y[:, i], nlpi)
            end
        end
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
        sys_opt::StateSpace{Discrete{T}, T}, K_opt::Matrix{T}, x0_opt, nlp = vec2modal(res.minimizer, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K, nnl)
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
        Astab = schur_stab(sys_opt.A, stable === true ? 0.01 : stable)
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
    e2, _ = pred ? lsim(prediction_error_filter(sysp_opt), pd) : lsim(sys_opt, d)
    R = cov(e2, dims = 2)
    mul!(sysp_opt.S, K_opt, R)
    Q0 = sysp_opt.S * K_opt'
    Q = Hermitian(Q0 + eps(maximum(abs, Q0)) * I)
    # Note: to get the correct Q so that kalman(sys, Q, R) = K, one must solve an LMI problem to minimize the cross term S. If one tolerates S, we have kalman(sys, Q, R, S) = K, where S = K*R. Without S, Q can be *very* far from correct.
    sysp_opt.R .= R
    sysp_opt.Q .= Q
    (; sys=sysp_opt, x0=x0_opt, res, nlp)
end

# this method fails for duals so this is an overload to silence the warning and save some time
ControlSystemsBase.balance_statespace(A::AbstractMatrix{<:ForwardDiff.Dual}, B::AbstractMatrix, C::AbstractMatrix, perm::Bool=false) = A,B,C,I

trivec(A) = [A[diagind(A, -1)]; A[diagind(A, 0)]; A[diagind(A, 1)]]

function vec2modal(p, ny, nu, nx, timeevol, zeroD, pred, D0, K0, nnl)
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
    x0 = p[ki[end]+1:end-nnl] # x0 may belong to an extended predictor state and can have any length
    nlpi = p[end-nnl+1:end]
    ss(A, B, C, D, timeevol), K, x0, nlpi
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
#         F = freqresp(sys, data.w)
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




# using ControlSystemsBase ControlSystemIdentification, Optim
# G = modal_form(ssrand(2,3,4))[1]
# w = exp10.(LinRange(-2, 2, 30))
# r = freqresp(G, w)
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
    if ControlSystemsBase.issiso(sysm)
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


"""
    nonlinear_pem(
        d::IdData,
        discrete_dynamics,
        measurement,
        p0,
        x0,
        R1,
        R2,
        nu;
        optimizer = LevenbergMarquardt(),
        λ = 1.0,
        optimize_x0 = true,
        kwargs...,
    )

Nonlinear Prediction-Error Method (PEM).

This method attempts to find the optimal vector of parameters, ``p``, and the initial condition ``x_0``, that minimizes the sum of squared one-step prediction errors. The prediction is performed using an Unscented Kalman Filter (UKF) and the optimization is performed using a Gauss-Newton method. 

!!! info "Requires LeastSquaresOptim.jl"
    This function is available only if LeastSquaresOptim.jl is manually installed and loaded by the user.

# Arguments:
- `d`: Identification data
- `discrete_dynamics`: A dynamics function `(xₖ, uₖ, p, t) -> x(k+1)` that takes the current state `x`, input `u`, parameters `p`, and time `t` and returns the next state `x(k+1)`.
- `measurement`: The measurement / output function of the nonlinear system `(xₖ, uₖ, p, t) -> yₖ`
- `p0`: The initial guess for the parameter vector
- `x0`: The initial guess for the initial condition
- `R1`: Dynamics noise covariance matrix (increasing this makes the algorithm trust the model less)
- `R2`: Measurement noise covariance matrix (increasing this makes the algorithm trust the measurements less)
- `nu`: Number of inputs to the system
- `optimizer`: Any optimizer from [LeastSquaresOptim](https://github.com/matthieugomez/LeastSquaresOptim.jl)
- `λ`: A weighting factor to minimize `dot(e, λ, e`). A commonly used metric is `λ = Diagonal(1 ./ (mag.^2))`, where `mag` is a vector of the "typical magnitude" of each output. Internally, the square root of `W = sqrt(λ)` is calculated so that the residuals stored in `res` are `W*e`.
- `optimize_x0`: Whether to optimize the initial condition `x0` or not. If `false`, the initial condition is fixed to the value of `x0` and the optimization is performed only on the parameters `p`.

The inner optimizer accepts a number of keyword arguments:
- `lower`: Lower bounds for the parameters and initial condition (if optimized). If `x0` is optimized, this is a vector with layout `[lower_p; lower_x0]`.
- `upper`: Upper bounds for the parameters and initial condition (if optimized). If `x0` is optimized, this is a vector with layout `[upper_p; upper_x0]`.
- `x_tol = 1e-8`
- `f_tol = 1e-8`
- `g_tol = 1e-8`
- `iterations = 1_000`
- `Δ = 10.0`
- `store_trace = false`

See [Identification of nonlinear models](https://baggepinnen.github.io/ControlSystemIdentification.jl/dev/nonlinear/) for more details.


!!! warning "Experimental"
    This function is considered experimental and may change in the future without respecting semantic versioning. This implementation also lacks a number of features associated with good nonlinear PEM implementations, such as regularization and support for multiple datasets.
"""
function nonlinear_pem end

# ==============================================================================
## Structured PEM
# ==============================================================================
using ForwardDiffChainRules
@ForwardDiff_frule LinearAlgebra.exp!(x1::AbstractMatrix{<:ForwardDiff.Dual}) true

function inner_constructor(p, constructor, Ts)
    sys = constructor(p)
    if iscontinuous(sys)
        return c2d(sys, Ts, :zoh), p.K, p.x0
    end
    return sys, p.K, p.x0
end

"""
    structured_pem(
        d,
        nx;
        focus = :prediction,
        p0,
        x0 = nothing,
        K0 = focus == :prediction ? zeros(nx, d.ny) : zeros(0,0),
        constructor,
        h = 1,
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
    )

Linear gray-box model identification using the prediction-error method (PEM).

This function differs from [`newpem`](@ref) in that here, the user controls the structure of the estimated model, while in `newpem` a generic black-box structure is used.

The user provides the function `constructor(p)` that constructs the model from the parameter vector `p`. This function must return a statespace system. `p0` is the corresponding initial guess for the parameters. `K0` is an initial guess for the observer gain (only used if `focus = :prediciton`) and `x0` is the initial guess for the initial condition (estimated automatically if not provided).
    
For other options, see [`newpem`](@ref).
"""
function structured_pem(
    d,
    nx;
    focus = :prediction,
    p0,
    x0 = nothing,
    K0 = focus == :prediction ? zeros(nx, d.ny) : zeros(0,0),
    constructor,
    h = 1,
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
) where {F, RE}
    T = promote_type(eltype(d.y), eltype(p0))
    nu = d.nu
    ny = d.ny
    pred = focus === :prediction
    pd = predictiondata(d)
    sys0 = constructor(p0)
    isdiscrete(sys0) || (sys0 = c2d(sys0, d.Ts))
    if x0 === nothing
        x0i::Vector{T} = if h == 1 || focus === :simulation
            T.(estimate_x0(sys0, d, min(length(d), 10nx)))
        else
            T.(estimate_x0(prediction_error_filter(PredictionStateSpace(sys0, K, 0, 0); h), pd, min(length(pd), 10nx)))
        end
    else
        length(x0) == nx || throw(ArgumentError("x0 must have length $nx"))
        x0i = x0
    end
    p0ca = ComponentArray(p = p0, K = T.(K0), x0 = x0i)
    function predloss(p)
        sysi, Ki, x0 = inner_constructor(p, constructor, d.Ts)
        syso = PredictionStateSpace(sysi, Ki, 0, 0)
        Pyh = ControlSystemsBase.observer_predictor(syso; h)
        yh, _ = lsim(Pyh, pd; x0)
        yh .= metric.(yh .- d.y)
        c1 = sum(yh)
        c1 + regularizer(p, syso)
    end
    function simloss(p)
        syssim, _, x0 = inner_constructor(p, constructor, d.Ts)
        y, _ = lsim(syssim, d; x0)
        y .= metric.(y .- d.y)
        sum(y) + regularizer(p, syssim)
    end
    local res, sys_opt, K_opt, x0_opt
    res = Optim.optimize(
        pred ? predloss : simloss,
        p0ca,
        optimizer,
        Optim.Options(;
            store_trace, show_trace, show_every, iterations, allow_f_increases,
            time_limit, x_tol, f_abstol, g_tol, f_calls_limit, g_calls_limit),
        autodiff = :forward,
    )
    sys_opt, K_opt, x0_opt = inner_constructor(res.minimizer, constructor, d.Ts)
    sysp_opt = PredictionStateSpace(
        sys_opt,
        pred ? K_opt : zeros(T, nx, ny),
        zeros(T, nx, nx),
        zeros(T, ny, ny),
        zeros(T, nx, ny),
    )
    pred && !isstable(observer_predictor(sysp_opt)) && @warn("Estimated predictor dynamics A-KC is unstable")
    (; sys=sysp_opt, x0=x0_opt, res)
end
