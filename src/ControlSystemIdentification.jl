module ControlSystemIdentification

export StateSpaceNoise, pem, simulation_errors, prediction_errors, predict, simulate

using DSP, LinearAlgebra, Statistics, Random, Optim, ControlSystems, FillArrays, Parameters

struct StateSpaceNoise{T, MT<:AbstractMatrix{T}} <: LTISystem
	A::MT
	B::MT
	K::MT
	Ts::Float64
	nx::Int
	nu::Int
	ny::Int
	function StateSpaceNoise(A::MT, B::MT, K::MT, Ts::Float64) where MT
		nx = size(A, 1)
		nu = size(B, 2)
		ny = size(K, 2)

		if size(A, 2) != nx && nx != 0
			error("A must be square")
		elseif size(B, 1) != nx
			error("B must have the same row size as A")
		elseif nx != size(K, 1)
			error("K must have the same row size as A")
		end

		# Validate sampling time
		if Ts < 0 && Ts != -1
			error("Ts must be either a positive number, 0
			(continuous system), or -1 (unspecified)")
		end
		new{eltype(A), typeof(A)}(A, B, K, Ts, nx, nu, ny)
	end
end

# Getter functions
ControlSystems.get_A(sys::StateSpaceNoise) = sys.A
ControlSystems.get_B(sys::StateSpaceNoise) = sys.B
ControlSystems.get_C(sys::StateSpaceNoise) = [I zeros(sys.ny,sys.nx-sys.ny)]
ControlSystems.get_D(sys::StateSpaceNoise) = zeros(sys.ny,sys.nu)

ControlSystems.get_Ts(sys::StateSpaceNoise) = sys.Ts

ControlSystems.ssdata(sys::StateSpaceNoise) = get_A(sys), get_B(sys), get_C(sys), get_D(sys)

# Funtions for number of intputs, outputs and states
ControlSystems.ninputs(sys::StateSpaceNoise) = sys.nu
ControlSystems.noutputs(sys::StateSpaceNoise) = sys.ny
ControlSystems.nstates(sys::StateSpaceNoise) = sys.nx

Base.ndims(::StateSpaceNoise) = 2 # NOTE: Also for SISO systems?
Base.size(sys::StateSpaceNoise) = (noutputs(sys), ninputs(sys)) # NOTE: or just size(get_D(sys))
Base.size(sys::StateSpaceNoise, d) = d <= 2 ? size(sys)[d] : 1
Base.eltype(::Type{S}) where {S<:StateSpaceNoise} = S
Base.convert(::Type{StateSpace}, sys::StateSpaceNoise) = ss(sys.A, sys.B, ControlSystems.get_C(sys), 0, sys.Ts)

function Base.getindex(sys::StateSpaceNoise, inds...)
	if size(inds, 1) != 2
		error("Must specify 2 indices to index statespace model")
	end
	rows, cols = ControlSystems.index2range(inds...) # FIXME: ControlSystems.index2range(inds...)
	return StateSpaceNoise(copy(sys.A), sys.B[:, cols], sys.K[:, rows], sys.Ts)
end

struct SysFilter{T<:StateSpaceNoise, FT}
	sys::T
	state::Vector{FT}
	yh::Vector{FT}
end
SysFilter(sys::LTISystem,x0=zeros(sys.nx)) = SysFilter(sys,x0,zeros(eltype(x0), sys.ny))

(s::SysFilter)(y, u) = sysfilter!(s.state, s.sys, y, u)
(s::SysFilter)(u) = sysfilter!(s.state, s.sys, u)

function sysfilter!(state::AbstractVector, sys::StateSpaceNoise, y, u)
	@unpack A,B,K = sys
	yh = state[1:length(y)] #vec(sys.C*state)
	e = y - yh
	state .= vec(sys.A*state + sys.B*u + sys.K*e)
	yh
end

function sysfilter!(state::AbstractVector, sys::StateSpaceNoise, u)
	@unpack A,B,K = sys
	yh = state[1:sys.ny] #vec(sys.C*state)
	state .= vec(sys.A*state + sys.B*u)
	yh
end

sysfilter!(s::SysFilter, y, u) = sysfilter!(s.state, s.sys, y, u)
sysfilter!(s::SysFilter, u) = sysfilter!(s.state, s.sys, u)

struct OberservationIterator{T}
	y::T
	u::T
end

observations(y,u) = OberservationIterator(y,u)

function Base.iterate(i::OberservationIterator{<:AbstractMatrix}, state=1)
	state > length(i) && return nothing
	((i.y[:,state],i.u[:,state]),state+1)
end
Base.length(i::OberservationIterator{<:AbstractMatrix}) = size(i.y, 2)

function Base.iterate(i::OberservationIterator{<:AbstractVector{<:Union{AbstractVector, Number}}}, state=1)
	state > length(i) && return nothing
	((i.y[state],i.u[state]),state+1)
end
Base.length(i::OberservationIterator{<:AbstractVector{<:Union{AbstractVector, Number}}}) = length(i.y)

struct PredictionErrorIterator{T}
	model
	oi::OberservationIterator{T}
end

prediction_errors(model, y, u) = PredictionErrorIterator(model, observations(y,u))

function Base.iterate(i::PredictionErrorIterator, state=1)
	state >= length(i) && return nothing
	(y,u), state1 = iterate(i.oi, state)
	yh = i.model(y,u)
	(y-yh,state1)
end
Base.length(i::PredictionErrorIterator) = length(i.oi)

mutable struct SimulationErrorIterator{T}
	model
	oi::OberservationIterator{T}
	yh # Stores the last prediction
end

simulation_errors(model, y, u) = SimulationErrorIterator(model, observations(y,u), zeros(obslength(y)))

function Base.iterate(i::SimulationErrorIterator, state=1)
	state >= length(i) && return nothing
	(y,u), state1 = iterate(i.oi, state)
	i.yh = i.model(i.yh,u)
	(y-i.yh,state1)
end
Base.length(i::SimulationErrorIterator) = length(i.oi)

obslength(y::AbstractMatrix) = size(y,1)
obslength(y::AbstractVector) = length(y[1])

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
- `solver` Defaults to `Optim.BFGS()`
- `kwargs`: additional keyword arguments are sent to `Optim.Options`.

# Return values
- `sys::StateSpaceNoise`: identified system. Can be converted to `StateSpace` by `convert(StateSpace, sys)`, but this will discard the Kalman gain matrix.
- `x0`: Estimated initial state
- `opt`: Optimization problem structure. Contains info of the result of the optimization problem
"""
function pem(y, u; nx, solver = BFGS(), focus=:prediction, metric=abs2, kwargs...)
	nu,ny = obslength(u),obslength(y)

	x0 = 0.001randn(nx)
	i0 = 0.001randn(ny)
	A = 0.0001randn(nx,ny)
	B = 0.001randn(nx,nu)
	K = 0.001randn(nx,ny)
	p = [A[:];B[:];K[:];x0]
	cf = p->pem_costfun(p,y,u,nx,metric)
	opt = optimize(cf, p, solver, Optim.Options(;iterations=200, kwargs...); autodiff = :forward)
	println(opt)
	if focus == :simulation
		@info "Focusing on simulation"
		cf = p->sem_costfun(p,y,u,nx,metric)
		opt = optimize(cf, Optim.minimizer(opt), NewtonTrustRegion(), Optim.Options(;iterations=100, kwargs...); autodiff = :forward)
		println(opt)
	end
	model = model_from_params(Optim.minimizer(opt), nx, ny, nu)
	model.sys, copy(model.state), opt
end


Base.oftype(x::Vector{<:Vector}, y::Vector{<:Vector}) = y
Base.oftype(x::Matrix, y::Vector{<:Vector}) = reduce(hcat,y)
Base.oftype(x::Matrix, y::Matrix) = y
Base.oftype(x::Vector{<:Vector}, y::Matrix) = [y[:,i] for i in 1:size(y,2)]
function predict(sys, y, u, x0=zeros(sys.nx))
	model = SysFilter(sys, copy(x0))
	yh = [model(yt,ut) for (yt,ut) in observations(y,u)]
	oftype(y,yh)
end

function simulate(sys, u, x0=zeros(sys.nx))
	model = SysFilter(sys, copy(x0))
	yh = map(observations(u,u)) do (ut,_)
		model(ut)
	end
	oftype(u,yh)
end

function ControlSystems.lsim(sys::StateSpaceNoise, u; x0=zeros(sys.nx))
	simulate(sys, u, x0)
end

end # module
