function mats(p)
    p.A, p.B, p.K
end
function model_from_params(p, h, ny)
    A, B, K = mats(p)
    x0      = copy(p.x0)
    sys     = StateSpaceNoise(A, B, K, h)
    sysf    = SysFilter(sys, x0, similar(x0, ny))
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
- `sys::StateSpaceNoise`: identified system. Can be converted to `StateSpace` by `convert(StateSpace, sys)`, but this will discard the Kalman gain matrix.
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
    solver              = BFGS(),
    focus               = :prediction,
    metric              = sse,
    regularizer         = p -> 0,
    iterations          = 1000,
    stabilize_predictor = true,
    difficult           = false,
    A                   = 0.0001randn(nx, nx),
    B                   = 0.001randn(nx, obslength(input(d))),
    # C                   = 0.001randn(obslength(output(d)), nx),
    K                   = 0.001randn(nx, obslength(output(d))),
    x0                  = 0.001randn(nx),
    kwargs...,
)

    y, u = output(d), input(d)
    nu, ny = obslength(u), obslength(y)
    if size(A,1) != size(A,2) # Old API
        A = [A zeros(nx, nx-ny)] 
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
    s            = model.sys
    @unpack A, K = s
    C            = s.C
    poles        = eigvals(A - K * C)
    newpoles     = map(poles) do p
        ap = abs(p)
        ap <= 1 && (return p)
        p / (ap + sqrt(eps()))
    end
    K2           = ControlSystems.acker(A', C', newpoles)' .|> real
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


using JuliaFormatter; JuliaFormatter.format_file(@__FILE__)

using Optim, Optim.LineSearches
function newpem(d, nx; zeroD = true, sys0 = subspaceid(d, nx; zeroD), focus = :prediction, optimizer = BFGS(alphaguess = LineSearches.InitialStatic(alpha=1), linesearch = LineSearches.HagerZhang()))
    nu = d.nu
    ny = d.ny
    A,B,C,D = ssdata(sys0)
    K = sys0.K
    pred = focus === :prediction
    pd = ControlSystemIdentification.predictiondata(d)
    if pred
        p0 = zeroD ? ComponentArray(; A, B, C, K) : ComponentArray(; A, B, C, D, K)
    else
        p0 = zeroD ? ComponentArray(; A, B, C) : ComponentArray(; A, B, C, D)
    end
    function predloss(p)
        # p0 .= p # write into already existing initial guess
        syso = ControlSystemIdentification.PredictionStateSpace(ss(p.A, p.B, p.C, zeroD ? 0 : p.D, d.timeevol), p.K, 0, 0)
        Pe = ControlSystemIdentification.prediction_error(syso)
        x0 = estimate_x0(Pe, pd, min(length(pd), 10nx))
        e,_ = lsim(Pe, pd; x0)
        mean(abs2, e)
    end
    function simloss(p)
        # p0 .= p # write into already existing initial guess
        syso = ss(p.A, p.B, p.C, zeroD ? 0 : p.D, d.timeevol)
        x0 = estimate_x0(syso, d, min(length(d), 10nx))
        e,_ = lsim(syso, d; x0)
        mean(abs2, e)
    end
    res = Optim.optimize(
        pred ? predloss : simloss,
        p0,
        optimizer,
        Optim.Options(
            store_trace       = true,
            show_trace        = true,
            show_every        = 50,
            iterations        = 10000,
            allow_f_increases = false,
            time_limit        = 100,
            x_tol             = 0,
            f_abstol          = 0,
            g_tol             = 1e-12,
            f_calls_limit     = 0,
            g_calls_limit     = 0,
        ),
        autodiff = :forward,
    )
    p = res.minimizer
    syso = ControlSystemIdentification.PredictionStateSpace(ss(p.A, p.B, p.C, zeroD ? 0 : p.D, d.timeevol), pred ? p.K : zeros(nx, d.ny), zeros(nx,nx), zeros(ny,ny))
    all(e->abs(e) < 1, eigvals(syso.A-syso.K*syso.C)) || @warn("Predictor A-KC unstable")
    Pe = ControlSystemIdentification.prediction_error(syso)
    e = lsim(Pe, pd)[1]
    R = cov(e, dims=2)
    @warn "K not updated after opt"
    Q = Hermitian(K*R*K' + eps()*I)
    # K = ((R+CXC')^(-1)(CXA'+S'))'
    # solve for X
    # solve for Q from  A'XA - X - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0
    @warn "probaby some error in Q matrix"
    syso.R .= R
    syso.Q .= Q
    syso, res
end