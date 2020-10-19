abstract type AbstractIdData end

const AnyInput = Union{AbstractArray,AbstractIdData}

struct InputOutputData{Y,U,T} <: AbstractIdData
    y::Y
    u::U
    Ts::T
end

struct OutputData{Y,T} <: AbstractIdData
    y::Y
    Ts::T
end

struct InputOutputStateData{Y,U,X,T} <: AbstractIdData
    y::Y
    u::U
    x::X
    Ts::T
end

autodim(x::Vector{<:AbstractVector}) = x
autodim(x::AbstractVector) = x
function autodim(x)
    r = size(x, 1)
    c = size(x, 2)
    if (c < 5 && c < r) || (r > 4c)
        @info "Transposing input. The convention used in ControlSystemIdentification is that input-output data is made out of either of 1) Vectors with scalars, 2) vectors of vectors or 3) matrices with time along the second dimension. The supplied input appears to be multidimensional and have time in the first dimension." maxlog =
            3
        return copy(x')
    end
    x
end

function Base.show(io::IO, d::OutputData)
    write(io, "Output data of length $(length(d)) with $(noutputs(d)) outputs")
end
function Base.show(io::IO, d::InputOutputData)
    write(
        io,
        "InputOutput data of length $(length(d)) with $(noutputs(d)) outputs and $(ninputs(d)) inputs",
    )
end


iddata(y::AbstractArray, Ts::Union{Real,Nothing} = nothing) = OutputData(autodim(y), Ts)
iddata(y::AbstractArray, u::AbstractArray, Ts::Union{Real,Nothing} = nothing) =
    InputOutputData(autodim(y), autodim(u), Ts)
iddata(
    y::AbstractArray,
    u::AbstractArray,
    x::AbstractArray,
    Ts::Union{Real,Nothing} = nothing,
) = InputOutputStateData(autodim(y), autodim(u), x, Ts)


output(d::AbstractIdData) = d.y
input(d::AbstractIdData) = d.u
LowLevelParticleFilters.state(d::AbstractIdData) = d.x
output(d::AbstractArray) = d
input(d::AbstractArray) = d
LowLevelParticleFilters.state(d::AbstractArray) = d
hasinput(::OutputData) = false
hasinput(::AbstractIdData) = true
hasinput(::AbstractArray) = true
hasinput(::ControlSystems.LTISystem) = true
ControlSystems.noutputs(d::AbstractIdData) = obslength(d.y)
ControlSystems.ninputs(d::AbstractIdData) = hasinput(d) ? obslength(d.u) : 0
ControlSystems.nstates(d::AbstractIdData) = 0
ControlSystems.nstates(d::InputOutputStateData) = obslength(d.x)
obslength(d::AbstractIdData) = ControlSystems.noutputs(d)
sampletime(d::AbstractIdData) = d.Ts === nothing ? 1.0 : d.Ts
function Base.length(d::AbstractIdData)
    y = output(d)
    y isa AbstractMatrix && return size(y, 2)
    return length(y)
end

function Base.getproperty(d::AbstractIdData, s::Symbol)
    if s === :fs || s === :Fs
        return 1 / d.Ts
    end
    return getfield(d, s)
end

function Base.:(==)(d1::T, d2::T) where {T<:AbstractIdData}
    all(fieldnames(T)) do field
        getfield(d1, field) == getfield(d2, field)
    end
end


timevec(d::AbstractIdData) = range(0, step = sampletime(d), length = length(d))


function apply_fun(fun, d::OutputData, Ts = d.Ts)
    iddata(fun(d.y), Ts)
end

"""
	apply_fun(fun, d::InputOutputData)

Apply `fun(y)` to all time series `y[,u,[x]] ∈ d` and return a new `iddata` with the transformed series.
"""
function apply_fun(fun, d::InputOutputData, Ts = d.Ts)
    iddata(fun(d.y), fun(d.u), Ts)
end

function apply_fun(fun, d::InputOutputStateData, Ts = d.Ts)
    iddata(fun(d.y), fun(d.u), fun(d.x), Ts)
end

function Base.getindex(d::Union{InputOutputData,InputOutputStateData}, i, j)
    iddata(d.y[i:i, :], d.u[j:j, :], d.Ts)
end

function Base.getindex(d::AbstractIdData, i)
    apply_fun(d) do y
        y[:, i]
    end
end

"""
dr = resample(d::InputOutputData, f)

Resample iddata `d` with fraction `f`, e.g., `f = fs_new / fs_original`.
"""
DSP.resample(d::AbstractIdData, f) = apply_fun(y -> resample(y, f), d, d.Ts / f)

function Base.hcat(d1::InputOutputData, d2::InputOutputData)
    @assert d1.Ts == d2.Ts
    iddata([d1.y d2.y], [d1.u d2.u], d1.Ts)
end

struct StateSpaceNoise{T,MT<:AbstractMatrix{T}} <:
       ControlSystems.AbstractStateSpace{Discrete{Float64}}
    A::MT
    B::MT
    K::MT
    Ts::Discrete{Float64}
    nx::Int
    nu::Int
    ny::Int
    function StateSpaceNoise(
        A::MT,
        B::MT,
        K::MT,
        Ts::Union{Real,Discrete{Float64}},
    ) where {MT}
        Ts = Ts isa Real ? Discrete(Float64(Ts)) : Float64(Ts)
        nx, nu, ny = ControlSystems.state_space_validation(
            A,
            B,
            K',
            zeros(size(K', 1), size(B, 2)),
            Ts,
        )
        new{eltype(A),typeof(A)}(A, B, K, Ts, nx, nu, ny)
    end
end

ControlSystems.isstable(s::StateSpaceNoise) =
    all(abs(e) <= 1 for e in eigvals(s.A - s.K * s.C))


# Funtions for number of intputs, outputs and states
# ControlSystems.ninputs(sys::StateSpaceNoise) = sys.nu
# ControlSystems.noutputs(sys::StateSpaceNoise) = sys.ny
# ControlSystems.nstates(sys::StateSpaceNoise) = sys.nx
#
# Base.ndims(::StateSpaceNoise) = 2 # NOTE: Also for SISO systems?
# Base.size(sys::StateSpaceNoise) = (noutputs(sys), ninputs(sys)) # NOTE: or just size(get_D(sys))
# Base.size(sys::StateSpaceNoise, d) = d <= 2 ? size(sys)[d] : 1
Base.eltype(::Type{S}) where {S<:StateSpaceNoise} = S
ControlSystems.numeric_type(::Type{<:StateSpaceNoise{T}}) where {T} = T
Base.convert(::Type{StateSpace}, sys::StateSpaceNoise) = ss(sys.A, sys.B, sys.C, 0, sys.Ts)
ControlSystems.ss(sys::StateSpaceNoise) = convert(StateSpace, sys)
ControlSystems.tf(sys::StateSpaceNoise) = tf(ss(sys))

function Base.getproperty(sys::StateSpaceNoise, p::Symbol)
    if p == :C
        return [I zeros(sys.ny, sys.nx - sys.ny)]
    elseif p == :D
        return zeros(sys.ny, sys.nu)
    end
    return getfield(sys, p)
end

ControlSystems.innovation_form(sys::StateSpaceNoise) =
    ss(sys.A, sys.K, sys.C, Matrix(Eye(sys.ny)), sys.Ts) # innovation model




function Base.getindex(sys::StateSpaceNoise, inds...)
    if size(inds, 1) != 2
        error("Must specify 2 indices to index statespace model")
    end
    rows, cols = ControlSystems.index2range(inds...) # FIXME: ControlSystems.index2range(inds...)
    return StateSpaceNoise(copy(sys.A), sys.B[:, cols], sys.K[:, rows], sys.Ts)
end

struct SysFilter{T<:AbstractStateSpace{<:Discrete},FT}
    sys::T
    state::Vector{FT}
    yh::Vector{FT}
end
SysFilter(sys::LTISystem, x0 = zeros(sys.nx)) =
    SysFilter(sys, x0, zeros(eltype(x0), sys.ny))

(s::SysFilter)(y, u) = sysfilter!(s.state, s.sys, y, u)
(s::SysFilter)(u) = sysfilter!(s.state, s.sys, u)
sysfilter!(s::SysFilter, y, u) = sysfilter!(s.state, s.sys, y, u)
sysfilter!(s::SysFilter, u) = sysfilter!(s.state, s.sys, u)

function sysfilter!(state::AbstractVector, sys::StateSpaceNoise, y, u)
    @unpack A, B, K, ny = sys
    yh = state[1:ny] #vec(sys.C*state)
    e = y .- yh
    state .= vec(A * state + B * u + K * e)
    yh
end

function sysfilter!(state::AbstractVector, sys::StateSpaceNoise, u)
    @unpack A, B, K, ny = sys
    yh = state[1:ny] #vec(C*state)
    state .= vec(A * state + B * u)
    yh
end

function sysfilter!(state::AbstractVector, sys::StateSpace, y, u)
    @unpack A, B, C, D = sys
    yh = vec(C * state + D * u)
    state .= vec(A * state + B * u)
    yh
end

function sysfilter!(state::AbstractVector, sys::StateSpace, u)
    @unpack A, B, C, D = sys
    yh = vec(C * state + D * u)
    state .= vec(A * state + B * u)
    yh
end


struct OberservationIterator{T}
    y::T
    u::T
end

observations(y, u) = OberservationIterator(y, u)

function Base.iterate(i::OberservationIterator{<:AbstractMatrix}, state = 1)
    state > length(i) && return nothing
    ((i.y[:, state], i.u[:, state]), state + 1)
end
Base.length(i::OberservationIterator{<:AbstractMatrix}) = size(i.y, 2)

function Base.iterate(
    i::OberservationIterator{<:AbstractVector{<:Union{AbstractVector,Number}}},
    state = 1,
)
    state > length(i) && return nothing
    ((i.y[state], i.u[state]), state + 1)
end
Base.length(i::OberservationIterator{<:AbstractVector{<:Union{AbstractVector,Number}}}) =
    length(i.y)

struct PredictionErrorIterator{T,MT}
    model::MT
    oi::OberservationIterator{T}
end

prediction_errors(model, y, u) = PredictionErrorIterator(model, observations(y, u))

function Base.iterate(i::PredictionErrorIterator, state = 1)
    state >= length(i) && return nothing
    (y, u), state1 = iterate(i.oi, state)
    yh = i.model(y, u)
    (y .- yh, state1)
end
Base.length(i::PredictionErrorIterator) = length(i.oi)

mutable struct SimulationErrorIterator{T,MT,YT}
    model::MT
    oi::OberservationIterator{T}
    yh::YT # Stores the last prediction
end

simulation_errors(model, y, u) = SimulationErrorIterator(
    model,
    observations(y, u),
    zeros(eltype(model.sys.A), obslength(y)),
)

function Base.iterate(i::SimulationErrorIterator, state = 1)
    state >= length(i) && return nothing
    (y, u), state1 = iterate(i.oi, state)
    i.yh = i.model(u)
    (y - i.yh, state1)
end
Base.length(i::SimulationErrorIterator) = length(i.oi)

@recipe function plot(d::AbstractIdData)
    y = time1(output(d))
    n = noutputs(d)
    if hasinput(d)
        u = time1(input(d))
        n += ninputs(d)
    end
    layout --> (n, 1)
    legend --> false
    xguide --> "Time"
    link --> :x
    xvec = range(0, step = sampletime(d), length = length(d))

    for i = 1:size(y, 2)
        @series begin
            title --> "Output $i"
            label --> "Output $i"
            xvec, y[:, i]
        end
    end
    if hasinput(d)
        for i = 1:size(u, 2)
            @series begin
                title --> "Input $i"
                label --> "Input $i"
                xvec, u[:, i]
            end
        end
    end
end
