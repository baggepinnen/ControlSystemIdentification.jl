
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

ControlSystems.isstable(s::StateSpaceNoise) = all(abs(e) <= 1 for e in eigvals(s.A-s.K*s.C))

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
Base.convert(::Type{StateSpace}, sys::StateSpaceNoise) = ss(sys.A, sys.B, sys.C, 0, sys.Ts)
ControlSystems.ss(sys::StateSpaceNoise) = convert(StateSpace,sys)
ControlSystems.tf(sys::StateSpaceNoise) = tf(ss(sys))

function Base.getproperty(sys::StateSpaceNoise, p::Symbol)
	if p == :C
		return [I zeros(sys.ny,sys.nx-sys.ny)]
	end
	return getfield(sys,p)
end

ControlSystems.innovation_form(sys::StateSpaceNoise) = ss(sys.A, sys.K, sys.C, Matrix(Eye(sys.ny)), sys.Ts) # innovation model




function Base.getindex(sys::StateSpaceNoise, inds...)
	if size(inds, 1) != 2
		error("Must specify 2 indices to index statespace model")
	end
	rows, cols = ControlSystems.index2range(inds...) # FIXME: ControlSystems.index2range(inds...)
	return StateSpaceNoise(copy(sys.A), sys.B[:, cols], sys.K[:, rows], sys.Ts)
end

struct SysFilter{T<:Union{StateSpaceNoise, StateSpace}, FT}
	sys::T
	state::Vector{FT}
	yh::Vector{FT}
end
SysFilter(sys::LTISystem,x0=zeros(sys.nx)) = SysFilter(sys,x0,zeros(eltype(x0), sys.ny))

(s::SysFilter)(y, u) = sysfilter!(s.state, s.sys, y, u)
(s::SysFilter)(u) = sysfilter!(s.state, s.sys, u)
sysfilter!(s::SysFilter, y, u) = sysfilter!(s.state, s.sys, y, u)
sysfilter!(s::SysFilter, u) = sysfilter!(s.state, s.sys, u)

function sysfilter!(state::AbstractVector, sys::StateSpaceNoise, y, u)
	@unpack A,B,K,ny = sys
	yh     = state[1:ny] #vec(sys.C*state)
	e      = y - yh
	state .= vec(A*state + B*u + K*e)
	yh
end

function sysfilter!(state::AbstractVector, sys::StateSpaceNoise, u)
	@unpack A,B,K,ny = sys
	yh     = state[1:ny] #vec(C*state)
	state .= vec(A*state + B*u)
	yh
end

function sysfilter!(state::AbstractVector, sys::StateSpace, y, u)
	@unpack A,B,C,D = sys
	yh     = vec(C*state + D*u)
	state .= vec(A*state + B*u)
	yh
end

function sysfilter!(state::AbstractVector, sys::StateSpace, u)
	@unpack A,B,C,D = sys
	yh     = vec(C*state + D*u)
	state .= vec(A*state + B*u)
	yh
end


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

struct PredictionErrorIterator{T,MT}
	model::MT
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

mutable struct SimulationErrorIterator{T,MT,YT}
	model::MT
	oi::OberservationIterator{T}
	yh::YT # Stores the last prediction
end

simulation_errors(model, y, u) = SimulationErrorIterator(model, observations(y,u), zeros(eltype(model.sys.A), obslength(y)))

function Base.iterate(i::SimulationErrorIterator, state=1)
	state >= length(i) && return nothing
	(y,u), state1 = iterate(i.oi, state)
	i.yh = i.model(u)
	(y-i.yh,state1)
end
Base.length(i::SimulationErrorIterator) = length(i.oi)
