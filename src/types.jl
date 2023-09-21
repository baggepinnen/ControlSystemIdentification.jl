import DSP.AbstractFFTs
import DSP.AbstractFFTs: fft

"""
See [`iddata`](@ref)
"""
abstract type AbstractIdData end

const AnyInput = Union{AbstractArray,AbstractIdData}

"""
See [`iddata`](@ref)
"""
struct InputOutputData{Y,U,T} <: AbstractIdData
    y::Y
    u::U
    Ts::T
end

"""
See [`iddata`](@ref)
"""
struct InputOutputFreqData{Y,U,W} <: AbstractIdData
    y::Y
    u::U
    w::W
end

"""
See [`iddata`](@ref)
"""
struct OutputData{Y,T} <: AbstractIdData
    y::Y
    Ts::T
end

"""
See [`iddata`](@ref)
"""
struct InputOutputStateData{Y,U,X,T} <: AbstractIdData
    y::Y
    u::U
    x::X
    Ts::T
end

autodim(x::Vector{<:AbstractVector}) = x
autodim(x::AbstractVector) = transpose(x)
function autodim(x)
    r = size(x, 1)
    c = size(x, 2)
    if (c < 5 && c < r) || (r > 4c)
        @info "Transposing input. The convention used in ControlSystemIdentification is that input-output data is made out of either of 1) Vectors with scalars, 2) vectors of vectors or 3) matrices with time along the second dimension. The supplied input appears to be multidimensional and have time in the first dimension." maxlog =
            3
        return copy(transpose(x))
    end
    x
end

function Base.show(io::IO, d::OutputData)
    write(io, "Output data of length $(length(d)) with $(noutputs(d)) outputs, Ts = $(d.Ts)")
end
function Base.show(io::IO, d::InputOutputData)
    write(
        io,
        "InputOutput data of length $(length(d)), $(noutputs(d)) outputs, $(ninputs(d)) inputs, Ts = $(d.Ts)",
    )
end


iddata(y::AbstractArray, Ts::Union{Real,Nothing} = nothing) = OutputData(autodim(y), Ts)
iddata(y::AbstractArray, u::AbstractArray, Ts::Union{Real,Nothing} = nothing) =
    InputOutputData(autodim(y), autodim(u), Ts)

"""
    iddata(y,       Ts = nothing)
    iddata(y, u,    Ts = nothing)
    iddata(y, u, x, Ts = nothing)

Create a **time-domain** identification data object. 

# Arguments
- `y::AbstractArray`: output data (required)
- `u::AbstractArray`: input data (if available)
- `x::AbstractArray`: state data (if available)
- `Ts::Union{Real,Nothing} = nothing`: optional sample time

If the time-series are multivariate, time is in the *last* dimension,
i.e., the sizes of the arrays are `(num_variables, num_timepoints)` (see examples below).

# Operations on iddata
- [`detrend`](@ref)
- [`prefilter`](@ref)
- [`resample`](@ref)
- append two along the time dimension `[d1 d2]` (only do this if the state of the system at the end of `d1` is close to the state at the beginning of `d2`)
- index time series `d[output_index, input_index]`
- index the time axis with indices `d[time_indices]`
- index the time axis with seconds `d[3Sec:12Sec]` (`using ControlSystemIdentification: Sec`)
- access number of inputs, outputs and sample time: `d.nu, d.ny, d.Ts`
- access the time time vector `d.t`
- premultiply to scale outputs `C * d`. Scaling the outputs of a multiple-output system to have roughly the same size is usually recommended before estimating a model in case they have different magnitudes.
- postmultiply to scale inputs `d * B`
- [`writedlm`](@ref)
- [`ramp_in`](@ref), [`ramp_out`](@ref)
- `plot`
- [`specplot`](@ref)
- [`crosscorplot`](@ref)

# Examples
```jldoctest
julia> iddata(randn(10))
Output data of length 10 with 1 outputs, Ts = nothing

julia> iddata(randn(10), randn(10), 1)
InputOutput data of length 10, 1 outputs, 1 inputs, Ts = 1

julia> d = iddata(randn(2, 10), randn(3, 10), 0.1)
InputOutput data of length 10, 2 outputs, 3 inputs, Ts = 0.1

julia> [d d] # Concatenate along time
InputOutput data of length 20, 2 outputs, 3 inputs, Ts = 0.1

julia> d[1:3]
InputOutput data of length 3, 2 outputs, 3 inputs, Ts = 0.1

julia> d.nu
3

julia> d.t # access time vector
0.0:0.1:0.9
```

# Use of multiple datasets
Some estimation methods support the use of multiple datasets to estimate a model. In this case, the datasets are provided as a vector of iddata objects. The methods that currently support this are:
- [`arx`](@ref)
- [`era`](@ref)

Several of the other estimation methods can be made to accept multiple datasets with minor modifications.

In some situations, multiple datasets can also be handled by concatenation. For this to be a good idea, the state of the system at the end of one data set must be close to the state at the beginning of the next, e.g., all experiments start and end at the same operating point.
"""
iddata(
    y::AbstractArray,
    u::AbstractArray,
    x::AbstractArray,
    Ts::Union{Real,Nothing} = nothing,
) = InputOutputStateData(autodim(y), autodim(u), x, Ts)

"""
    iddata(y::AbstractArray, u::AbstractArray, w::AbstractVector)

Create a **frequency-domain** input-output data object. `w` is expected to be in rad/s.
"""
iddata(y::AbstractArray, u::AbstractArray, w::AbstractVector) = InputOutputFreqData(autodim(y), autodim(u), w)

"""
    iddata(res::ControlSystemsBase.SimResult)

Create an identification-data object directly from a simulation result.
"""
iddata(res::ControlSystemsBase.SimResult) = iddata(res.y, res.u, res.t[2]-res.t[1])


output(d::AbstractIdData)                        = d.y
input(d::AbstractIdData)                         = d.u
LowLevelParticleFilters.state(d::AbstractIdData) = d.x
output(d::AbstractArray)                         = d
input(d::AbstractArray)                          = d
LowLevelParticleFilters.state(d::AbstractArray)  = d
hasinput(::OutputData)                           = false
hasinput(::AbstractIdData)                       = true
hasinput(::AbstractArray)                        = true
hasinput(::ControlSystemsBase.LTISystem)             = true
ControlSystemsBase.noutputs(d::AbstractIdData)       = obslength(getfield(d, :y))
ControlSystemsBase.ninputs(d::AbstractIdData)        = hasinput(d) ? obslength(getfield(d, :u)) : 0
ControlSystemsBase.nstates(d::AbstractIdData)        = 0
ControlSystemsBase.nstates(d::InputOutputStateData)  = obslength(getfield(d, :x))
obslength(d::AbstractIdData)                     = ControlSystemsBase.noutputs(d)
sampletime(d::AbstractIdData)                    = d.Ts === nothing ? 1.0 : d.Ts
function Base.length(d::AbstractIdData)
    y = output(d)
    y isa AbstractMatrix && return size(y, 2)
    return length(y)
end

Base.axes(d::AbstractIdData, i::Integer) = Base.OneTo(i == 1 ? d.ny : d.nu)

Base.lastindex(d::AbstractIdData) = length(d)

function w2Ts(w)
    N = length(w)
    N*maximum(w)/(2π*(N-1))
end

function Base.getproperty(d::AbstractIdData, s::Symbol)
    if s === :fs || s === :Fs
        return 1 / d.Ts
    elseif s === :Ts
        if d isa InputOutputFreqData
            d.w isa AbstractRange || error("Sample time is only aviable from a InputOutputFreqData if the frequency vector is an AbstractRange")
            N = length(d)
            return w2Ts(getfield(d, :w))
        else
            return getfield(d, :Ts)
        end
    elseif s === :timeevol
        return Discrete(d.Ts)
    elseif s === :t
        return timevec(d)
    elseif s === :w
        return d isa InputOutputFreqData ? getfield(d,:w) : (2π/length(d)).*timevec(d)
    elseif s === :f
        return d isa InputOutputFreqData ? getfield(d,:w)./(2π) : (1/length(d)).*timevec(d)
    elseif s === :ny
        return noutputs(d)
    elseif s === :nu
        return ninputs(d)
    elseif s === :nx
        return nstates(d)
    end
    return getfield(d, s)
end

function Base.:(==)(d1::T, d2::T) where {T<:AbstractIdData}
    all(fieldnames(T)) do field
        getfield(d1, field) == getfield(d2, field)
    end
end


timevec(d::AbstractIdData) = range(0, step = sampletime(d), length = length(d))
timevec(d::AbstractVector, h::Real) = range(0, step = h, length = length(d))
timevec(d::AbstractMatrix, h::Real) = range(0, step = h, length = maximum(size(d)))
function timevec(d::InputOutputFreqData)
    

end


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

torange(x::Number) = x:x
torange(x) = x

function Base.getindex(d::Union{InputOutputData,InputOutputStateData}, i, j)
    iddata(d.y[torange(i), :], d.u[torange(j), :], d.Ts)
end


function Base.getindex(d::AbstractIdData, i)
    apply_fun(d) do y
        y[:, i]
    end
end

function Base.getindex(d::AbstractIdData, i::AbstractRange)
    apply_fun(d, d.Ts === nothing ? d.Ts : d.Ts*step(i)) do y
        y[:, i]
    end
end

struct Sec <: Number
    i::Any
end
Base.:*(i, ::Type{Sec}) = Sec(i)
(::Colon)(start::Sec, stop::Sec) = (start, stop)

function Base.getindex(d::AbstractIdData, r::Tuple{Sec,Sec})
    t = timevec(d)
    s = findfirst(t .>= r[1].i)
    e = findlast(t .<= r[2].i)
    d[s:e]
end

function Base.:(*)(d::AbstractIdData, x)
    y,u = d.y, d.u
    iddata(y, x*u, d.Ts)
end

function Base.:(*)(x, d::AbstractIdData)
    y,u = d.y, d.u
    iddata(x*y, u, d.Ts)
end

"""
    dr = resample(d::InputOutputData, f)

Resample iddata `d` with fraction `f`, e.g., `f = fs_new / fs_original`.
"""
function DSP.resample(d::AbstractIdData, f)
    Ts = d.Ts === nothing ? 1.0 : d.Ts
    apply_fun(d, Ts / f) do y
        yr = mapslices(y, dims = 2) do y
            DSP.resample(y, f)
        end
        yr
    end
end

function DSP.resample(M::AbstractMatrix, f)
    mapslices(M, dims = 1) do y
        DSP.resample(y, f)
    end
end

"""
    detrend(d::AbstractArray)
    detrend(d::AbstractIdData)

Remove the mean from `d`.
"""
detrend(x::AbstractVector) = x .- mean(x)
detrend(x::AbstractMatrix) = x .- mean(x, dims=2)
detrend(d::AbstractIdData) = apply_fun(detrend, d)

function AbstractFFTs.fft(d::InputOutputData)
    y,u = time2(output(d)), time2(input(d))
    sN = √length(d)
    InputOutputFreqData(fft(y, 2) ./ sN, fft(u, 2) ./ sN, d.w)
end



function Base.hcat(d1::InputOutputData, d2::InputOutputData)
    @assert d1.Ts == d2.Ts
    iddata([d1.y d2.y], [d1.u d2.u], d1.Ts)
end

"""
    DelimitedFiles.writedlm(io::IO, d::AbstractIdData, args...; kwargs...)

Write identification data to disk.
"""
function DelimitedFiles.writedlm(io::IO, d::AbstractIdData, args...; kwargs...)
    writedlm(io, transpose([d.y; d.u]), args...; kwargs...)
end

"""
    ramp_in(d::InputOutputData, h::Int; rev = false)

Multiply the initial `h` samples of input and output signals with a linearly increasing ramp.
"""
function ramp_in(d::InputOutputData, h::Int; rev=false)
    if h <= 1
        return d
    end
    u,y = input(d), output(d)
    if rev
        ramp = [
            ones(length(d)-h)
            range(1, stop=0, length=h);
        ]
    else
        ramp = [
            range(0, stop=1, length=h);
            ones(length(d)-h)
        ]
    end
    u = u .* ramp'
    y = y .* ramp'
    iddata(y,u,d.Ts)
end

"""
    ramp_out(d::InputOutputData, h::Int)

Multiply the final `h` samples of input and output signals with a linearly decreasing  ramp.
"""
ramp_out(d::InputOutputData, h::Int) = ramp_in(d,h; rev=true)


## State space types ===========================================================

abstract type AbstractPredictionStateSpace{T} <: AbstractStateSpace{T} end

Base.@kwdef struct PredictionStateSpace{T, ST <: AbstractStateSpace{T}, KT, QT, RT, ST2} <: AbstractPredictionStateSpace{T}
# has at least K, but perhaps also covariance matrices? Would be nice in order to be able to resample he system. Can be nothing in case they are not known
    sys::ST
    K::KT
    Q::QT = nothing
    R::RT = nothing
    S::ST2 = nothing
    PredictionStateSpace(sys, K, Q=nothing, R=nothing, S=nothing) = new{typeof(sys.timeevol), typeof(sys),typeof(K),typeof(Q),typeof(R),typeof(S)}(sys, K, Q, R, S)
end


"""
    PredictionStateSpace{T, ST <: AbstractStateSpace{T}, KT, QT, RT, ST2} <: AbstractPredictionStateSpace{T}
    PredictionStateSpace(sys, K, Q=nothing, R=nothing, S=nothing)

A statespace type that contains an additional Kalman-filter model for prediction purposes.

# Arguments:
- `sys`: DESCRIPTION
- `K`: Infinite-horizon Kalman gain
- `Q = nothing`: Dynamics covariance
- `R = nothing`: Measurement covariance
- `S = nothing`: Cross-covariance
"""
PredictionStateSpace

Base.promote_rule(::Type{AbstractStateSpace{T}}, ::Type{<:AbstractPredictionStateSpace{T}}) where T<:ControlSystemsBase.TimeEvolution  = StateSpace{T<:ControlSystemsBase.TimeEvolution}

Base.promote_rule(::Type{StateSpace{T,F}}, ::Type{<:AbstractPredictionStateSpace{T}}) where {T<:ControlSystemsBase.TimeEvolution, F} = StateSpace{T, F}

Base.promote_rule(::Type{StateSpace{T,F}}, ::Type{PredictionStateSpace{T}}) where {T<:ControlSystemsBase.TimeEvolution, F} = StateSpace{T, F}

Base.convert(::Type{<:StateSpace{T}}, s::AbstractPredictionStateSpace{T}) where T<:ControlSystemsBase.TimeEvolution = deepcopy(s.sys)

function Base.:(-)(sys0::ST) where ST <: AbstractPredictionStateSpace
    otherfields = ntuple(i->getfield(sys0, i+1), fieldcount(ST)-1)
    sys = sys0.sys
    ST(typeof(sys)(sys.A, sys.B, -sys.C, -sys.D, sys.timeevol), otherfields...)
end

"""
    N4SIDStateSpace <: AbstractPredictionStateSpace
    
The result of statespace model estimation using the `n4sid` method.

# Fields:
- `sys`: estimated model in the form of a `StateSpace` object
- `Q`: estimated covariance matrix of the states
- `R`: estimated covariance matrix of the measurements
- `S`: estimated cross covariance matrix between states and measurements
- `K`: Kalman observer gain
- `P`: solution to the Riccatti equation
- `x`: estimated state trajectory (`n4sid`) or initial condition (`subspaceid`)
- `s`: singular value decomposition
- `fve`: Fraction of variance explained by singular values
"""
struct N4SIDStateSpace{Tsys,TQ,TR,TS,TK,TP,Tx,Ts,Tfve} <: AbstractPredictionStateSpace{Discrete{Float64}}
    sys::Tsys
    Q::TQ
    R::TR
    S::TS
    K::TK
    P::TP
    x::Tx
    s::Ts
    fve::Tfve
end

@inline function Base.getproperty(res::AbstractPredictionStateSpace, p::Symbol)
    if p ∈ (:A, :B, :C, :D, :nx, :ny, :nu, :Ts, :timeevol)
        return getproperty(getfield(res, :sys), p)
    end
    return getfield(res, p)
end

function Base.getindex(sys::AbstractPredictionStateSpace, inds...)
    if size(inds, 1) != 2
        error("Must specify 2 indices to index statespace model")
    end
    rows, cols = ControlSystemsBase.index2range(inds...) 
    return PredictionStateSpace(ss(copy(sys.A), sys.B[:, cols], sys.C[rows, :], sys.D[rows, cols], sys.timeevol), sys.K[:, rows], sys.Q, sys.R[rows, rows])
end

ControlSystemsBase.numeric_type(s::AbstractPredictionStateSpace) = ControlSystemsBase.numeric_type(s.sys) 


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
