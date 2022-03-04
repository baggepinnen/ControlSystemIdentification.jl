function mats(p)
    p.A, p.B, p.K
end
function model_from_params(p, h, ny)
    A, B, K = mats(p)
    x0 = copy(p.x0)
    sys = StateSpaceNoise(A, B, K, h)
    sysf = SysFilter(sys, x0, similar(x0, ny))
end

function pem_costfun(p, y, u, h, metric::F) where {F} # To ensure specialization on metric
    nu, ny = obslength(u), obslength(y)
    model = model_from_params(p, h, ny)
    return mean(metric(e) for e in prediction_errors(model, y, u))
end
function sem_costfun(p, y, u, h, metric::F) where {F} # To ensure specialization on metric
    nu, ny = obslength(u), obslength(y)
    model = model_from_params(p, h, ny)
    return mean(metric(e) for e in simulation_errors(model, y, u))
end


"""
    sys, x0, opt = pem(data; nx, kwargs...)

System identification using the prediction-error method.

# Arguments:
- `data`: iddata object containing `y` and `u`.
    - `y`: Measurements, either a matrix with time along dim 2, or a vector of vectors
    - `u`: Control signals, same structure as `y`
- `nx`: Number of poles in the estimated system. Thus number should be chosen as number of system poles plus number of poles in noise models for measurement noise and load disturbances.
- `focus`: Either `:prediction` or `:simulation`. If `:simulation` is chosen, a two stage problem is solved with prediction focus first, followed by a refinement for simulation focus.
- `metric`: A Function determining how the size of the residuals is measured, default `sse` (e'e), but any Function such as `norm`, `e->sum(abs,e)` or `e -> e'Q*e` could be used.
- `regularizer(p)=0`: function for regularization. The structure of `p` is detailed below
- `solver` Defaults to `Optim.BFGS()`
- `stabilize_predictor=true`: Modifies the estimated Kalman gain `K` in case `A-KC` is not stable by moving all unstable eigenvalues to the unit circle.
- `difficult=false`: If the identification problem appears to be difficult and ends up in a local minimum, set this flag to true to solve an initial global optimization problem to supply a good initial guess. This is expected to take some time.
- `kwargs`: additional keyword arguments are sent to `Optim.Options`.

# Return values
- `sys::StateSpaceNoise`: identified system. Can be converted to `StateSpace` by `convert(StateSpace, sys)` or `ss(sys)`, but this will discard the Kalman gain matrix, see `innovation_form` to obtain a predictor system.
- `x0`: Estimated initial state
- `opt`: Optimization problem structure. Contains info of the result of the optimization problem


## Structure of parameter vector `p`
The parameter vector is of type [`ComponentVector`](https://github.com/jonniedie/ComponentArrays.jl) and the fields `A,B,K,x0` can be accessed as `p.A` etc. The internal storage is according to
```julia
A = size(nx,nx)
B = size(nx,nu)
K = size(nx,ny)
x0 = size(nx)
p = [A[:]; B[:]; K[:]; x0]
```
"""
function pem(
    d;
    nx,
    solver = BFGS(),
    focus = :prediction,
    metric = sse,
    regularizer = p -> 0,
    iterations = 1000,
    stabilize_predictor = true,
    difficult = false,
    A = 0.0001randn(nx, nx),
    B = 0.001randn(nx, obslength(input(d))),
    # C                   = 0.001randn(obslength(output(d)), nx),
    K = 0.001randn(nx, obslength(output(d))),
    x0 = 0.001randn(nx),
    kwargs...,
)

    y, u = output(d), input(d)
    nu, ny = obslength(u), obslength(y)
    if size(A, 1) != size(A, 2) # Old API
        A = [A zeros(nx, nx - ny)]
    end
    p = ComponentVector((; A, B, K, x0))
    options = Options(; iterations = iterations, kwargs...)
    cost_pred = p -> pem_costfun(p, y, u, d.Ts, metric) + regularizer(p)
    if difficult
        stabilizer = p -> 10000 * !stabfun(d.Ts, ny)(p)
        options0 = Options(; iterations = iterations, kwargs...)
        opt = optimize(
            p -> cost_pred(p) + stabilizer(p),
            1000p,
            # ParticleSwarm(n_particles = 100length(p)),
            NelderMead(),
            options0,
        )
        p = minimizer(opt)
    end
    opt = optimize(cost_pred, p, solver, options; autodiff = :forward)
    println(opt)
    if focus == :simulation
        @info "Focusing on simulation"
        cost_sim = p -> sem_costfun(p, y, u, d.Ts, metric) + regularizer(p)
        opt = optimize(
            cost_sim,
            minimizer(opt),
            NewtonTrustRegion(),
            Options(; iterations = iterations ÷ 2, kwargs...);
            autodiff = :forward,
        )
        println(opt)
    end
    model = model_from_params(minimizer(opt), d.Ts, ny)
    if !isstable(model.sys)
        @warn("Estimated system does not have a stable prediction filter (A-KC)")
        if stabilize_predictor
            @info("Stabilizing predictor")
            # model = stabilize(model)
            model = stabilize(model, solver, options, cost_pred)
        end
    end
    isstable(ss(model.sys)) || @warn("Estimated system is not stable")

    model.sys, copy(model.state), opt
end

function stabilize(model)
    s = model.sys
    @unpack A, K = s
    C = s.C
    poles = eigvals(A - K * C)
    newpoles = map(poles) do p
        ap = abs(p)
        ap <= 1 && (return p)
        p / (ap + sqrt(eps()))
    end
    K2 = ControlSystems.acker(A', C', newpoles)' .|> real
    all(abs(p) <= 1 for p in eigvals(A - K * C)) || @warn("Failed to stabilize predictor")
    s.K .= K2
    model
end

function stabilize(model, solver, options, cfi)
    s = model.sys
    cost = function (p)
        maximum(abs.(eigvals(s.A - p * s.K * s.C))) - 0.9999999
    end
    p = fzero(cost, 1e-9, 1 - 1e-9)
    s.K .*= p
    model
end

function stabfun(h, ny)
    function (p)
        model = model_from_params(p, h, ny)
        isstable(model.sys)
    end
end


using Optim, Optim.LineSearches

"""
    newpem(
        d,
        nx;
        zeroD = true,
        sys0 = subspaceid(d, nx; zeroD),
        focus = :prediction,
        optimizer = BFGS(
            alphaguess = LineSearches.InitialStatic(alpha = 1),
            linesearch = LineSearches.HagerZhang(),
        ),
        zerox0 = false,
        initx0 = false,
        store_trace = true,
        show_trace = true,
        show_every = 50,
        iterations = 10000,
        allow_f_increases = false,
        time_limit = 100,
        x_tol = 0,
        f_abstol = 0,
        g_tol = 1e-12,
        f_calls_limit = 0,
        g_calls_limit = 0,
        metric::F = abs2,
    )

A new implementation of the prediction-error method (PEM). Note that this is an experimental implementation and subject to breaking changes not respecting semver.

# Arguments:
- `d`: [`iddata`](@ref)
- `nx`: Model order
- `zeroD`: Force zero `D` matrix
- `sys0`: Initial guess, if non provided, [`subspaceid`](@ref) is used as initial guess.
- `focus`: `prediction` or `:simulation`. If `:simulation`, hte `K` matrix will be zero.
- `optimizer`: One of Optim's optimizers
- `zerox0`: Force initial state to zero.
- `initx0`: Estimate initial state once, otherwise at each iteration
- `metric`: The metric used to measure residuals. Try, e.g., `abs` for better resistance to outliers.
The rest of the arguments are related to `Optim.Options`.
"""
function newpem(
    d,
    nx;
    zeroD = true,
    focus = :prediction,
    sys0 = subspaceid(d, nx; zeroD, focus),
    optimizer = BFGS(
        # alphaguess = LineSearches.InitialStatic(alpha = 0.95),
        linesearch = LineSearches.HagerZhang(),
    ),
    zerox0 = false,
    initx0 = false,
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
    metric::F = abs2,
) where F
    T = promote_type(eltype(d.y), eltype(sys0.A))
    nu = d.nu
    ny = d.ny
    ny <= nx || throw(ArgumentError("ny > nx not supported by this method."))
    sys0, Tmodal = modal_form(sys0)
    A, B, C, D = ssdata(sys0)
    K = if hasfield(typeof(sys0), :K)
        convert(Matrix{T}, sys0.K)
    else
        zeros(T, nx, ny)
    end
    pred = focus === :prediction
    pd = predictiondata(d)::typeof(d)
    p0::Vector{T} = if pred
        T.(zeroD ? [trivec(A); vec(B); vec(C); vec(K)] : [trivec(A); vec(B); vec(C); vec(D); vec(K)])
    else
        T.(zeroD ? [trivec(A); vec(B); vec(C)] : [trivec(A); vec(B); vec(C); vec(D)])
    end
    if zeroD
        D0 = zeros(T, ny, nu)
    end
    if initx0
        x0i = T.(estimate_x0(sys0, d, min(length(pd), 10nx)))
    end
    function predloss(p)
        sysi, Ki = vec2modal(p, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K)
        syso = PredictionStateSpace(sysi, Ki, 0, 0)
        Pe = prediction_error(syso)
        x0::Vector{eltype(p)} = initx0 ? x0i : zerox0 ? zeros(eltype(p), Pe.nx) : estimate_x0(Pe, pd, min(length(pd), 10nx))
        e, _ = lsim(Pe, pd; x0)
        sum(metric, e) #+ 1e-5*sum(abs2, p - p0)
    end
    function simloss(p)
        syssim, _ = vec2modal(p, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K)
        x0sim = initx0 ? x0i : zerox0 ? zeros(T, syssim.nx) : estimate_x0(syssim, d, min(length(d), 10nx))
        y, _ = lsim(syssim, d; x0=x0sim)
        y .= metric.(y .- d.y)
        sum(y)
    end
    res = Optim.optimize(
        pred ? predloss : simloss,
        p0,
        optimizer,
        Optim.Options(;
            store_trace, show_trace, show_every, iterations, allow_f_increases,
            time_limit, x_tol, f_abstol, g_tol, f_calls_limit, g_calls_limit),
        autodiff = :forward,
    )
    sys_opt::StateSpace{Discrete{Int64}, T}, K_opt::Matrix{T} = vec2modal(res.minimizer, ny, nu, nx, sys0.timeevol, zeroD, pred, D0, K)
    sysp_opt = PredictionStateSpace(
        sys_opt,
        pred ? K_opt : zeros(T, nx, ny),
        zeros(T, nx, nx),
        zeros(T, ny, ny),
    )
    all(e -> abs(e) < 1, eigvals(sysp_opt.A - sysp_opt.K * sysp_opt.C)) ||
        @warn("Predictor A-KC unstable")
    Pe = prediction_error(sysp_opt)
    e = lsim(Pe, pd)[1]
    R = cov(e, dims = 2)
    # @warn "K not updated after opt"
    Q = Hermitian(K_opt * R * K_opt' + eps() * I)
    # K = ((R+CXC')^(-1)(CXA'+S'))'
    # solve for X
    # solve for Q from  A'XA - X - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0
    # @warn "probaby some error in Q matrix"
    sysp_opt.R .= R
    sysp_opt.Q .= Q
    sysp_opt, res
end

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
        K = K0
    end
    ss(A, B, C, D, timeevol), K
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