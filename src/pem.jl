
function mats(p, nx, ny, nu)
	A         = zeros(eltype(p), nx, nx)
	for i = ny:nx-1
		A[i-ny+1,i+1] = 1
	end
	s,e       = 1,nx*ny
	A[:,1:ny] = reshape(p[s:e], nx, ny)
	s,e       = e+1,e+nu*nx
	B         = reshape(p[s:e], nx, nu)
	s,e       = e+1,e+ny*nx
	K         = reshape(p[s:e], nx, ny)
	A,B,K
end
function model_from_params(p, nx, ny, nu)
	A,B,K = mats(p, nx, ny, nu)
	x0    = copy(p[end-nx+1:end])
	sys   = StateSpaceNoise(A,B,K,1.)
	sysf  = SysFilter(sys,x0,similar(x0,ny))
end

function pem_costfun(p,y,u,nx,metric::F) where F # To ensure specialization on metric
	nu,ny = obslength(u),obslength(y)
	model = model_from_params(p, nx, ny, nu)
	return mean(metric(e) for e in prediction_errors(model,y,u))
end
function sem_costfun(p,y,u,nx,metric::F) where F # To ensure specialization on metric
	nu,ny = obslength(u),obslength(y)
	model = model_from_params(p, nx, ny, nu)
	return mean(metric(e) for e in simulation_errors(model,y,u))
end


"""
sys, x0, opt = pem(y, u; nx, kwargs...)

System identification using the prediction error method.

# Arguments:
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

## Structure of parameter vecotr `p`
```julia
A = size(nx,ny)
B = size(nx,nu)
K = size(nx,ny)
x0 = size(nx)
p = [A[:];B[:];K[:];x0]
```
"""
function pem(y, u; nx, solver = BFGS(), focus=:prediction, metric=sse, regularizer=p->0, iterations=100, stabilize_predictor=true, difficult=false, kwargs...)
	nu,ny = obslength(u),obslength(y)

	A       = 0.0001randn(nx,ny)
	B       = 0.001randn(nx,nu)
	K       = 0.001randn(nx,ny)
	x0      = 0.001randn(nx)
	p       = [A[:];B[:];K[:];x0]
	options = Options(;iterations=iterations, kwargs...)
	cfp     = p->pem_costfun(p,y,u,nx,metric) + regularizer(p)
	if difficult
		stabilizer = p-> 10000*!stabfun(nx,nu,ny)(p)
		options0 = Options(;iterations=iterations, kwargs...)
		opt = optimize(p->cfp(p)+stabilizer(p), 1000p, ParticleSwarm(n_particles=100length(p)), options0)
		p = minimizer(opt)
	end
	opt     = optimize(cfp, p, solver, options; autodiff = :forward)
	println(opt)
	if focus == :simulation
		@info "Focusing on simulation"
		cf = p->sem_costfun(p,y,u,nx,metric) + regularizer(p)
		opt = optimize(cf, minimizer(opt), NewtonTrustRegion(), Options(;iterations=iterations÷2, kwargs...); autodiff = :forward)
		println(opt)
	end
	model = model_from_params(minimizer(opt), nx, ny, nu)
	if !isstable(model.sys)
		@warn("Estimated system does not have a stable prediction filter (A-KC)")
		if stabilize_predictor
			@info("Stabilizing predictor")
			# model = stabilize(model)
			model = stabilize(model, solver, options, cfp)
		end
	end
	isstable(ss(model.sys)) || @warn("Estimated system is not stable")

	model.sys, copy(model.state), opt
end

function stabilize(model)
	s           = model.sys
	@unpack A,K = s
	C           = s.C
	poles       = eigvals(A-K*C)
	newpoles = map(poles) do p
		ap = abs(p)
		ap <= 1 && (return p)
		p / (ap + sqrt(eps()))
	end
	K2   = ControlSystems.acker(A',C', newpoles)' .|> real
	all(abs(p) <= 1 for p in eigvals(A-K*C)) || @warn("Failed to stabilize predictor")
	s.K .= K2
	model
end

function stabilize(model, solver, options, cfi)
	s = model.sys
	cost = function(p)
		maximum(abs.(eigvals(s.A-p*s.K*s.C)))-0.9999999
	end
	p = fzero(cost,1e-9,1-1e-9)
	s.K .*= p
	model
end

function stabfun(nx,ny,nu)
	function (p)
		model = model_from_params(p,nx,nu,ny)
		isstable(model.sys)
	end
end
