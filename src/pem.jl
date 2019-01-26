
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
	return sum(sum(metric,e) for e in prediction_errors(model,y,u))
end
function sem_costfun(p,y,u,nx,metric::F) where F # To ensure specialization on metric
	nu,ny = obslength(u),obslength(y)
	model = model_from_params(p, nx, ny, nu)
	return sum(sum(metric,e) for e in simulation_errors(model,y,u))
end


"""
sys, x0, opt = pem(y, u; nx, kwargs...)

System identification using the prediction error method.

# Arguments:
- `y`: Measurements, either a matrix with time along dim 2, or a vector of vectors
- `u`: Control signals, same structure as `y`
- `nx`: Number of poles in the estimated system. Thus number should be chosen as number of system poles plus number of poles in noise models for measurement noise and load disturbances.
- `focus`: Either `:prediction` or `:simulation`. If `:simulation` is chosen, a two stage problem is solved with prediction focus first, followed by a refinement for simulation focus.
- `metric`: A Function determining how the size of the residuals is measured, default `abs2`, but any Function such as `abs` or `x -> x'Q*x` could be used.
- `regularizer(p)=0`: function for regularization. The structure of `p` is detailed below
- `solver` Defaults to `Optim.BFGS()`
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
function pem(y, u; nx, solver = BFGS(), focus=:prediction, metric=abs2, regularizer=p->0, kwargs...)
	nu,ny = obslength(u),obslength(y)

	A = 0.0001randn(nx,ny)
	B = 0.001randn(nx,nu)
	K = 0.001randn(nx,ny)
	x0 = 0.001randn(nx)
	p = [A[:];B[:];K[:];x0]
	cf = p->pem_costfun(p,y,u,nx,metric) + regularizer(p)
	opt = optimize(cf, p, solver, Optim.Options(;iterations=200, kwargs...); autodiff = :forward)
	println(opt)
	if focus == :simulation
		@info "Focusing on simulation"
		cf = p->sem_costfun(p,y,u,nx,metric) + regularizer(p)
		opt = optimize(cf, Optim.minimizer(opt), NewtonTrustRegion(), Optim.Options(;iterations=100, kwargs...); autodiff = :forward)
		println(opt)
	end
	model = model_from_params(Optim.minimizer(opt), nx, ny, nu)
	model.sys, copy(model.state), opt
end
