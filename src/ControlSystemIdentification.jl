module ControlSystemIdentification

using ComponentArrays,
    ControlSystemsBase,
    DelimitedFiles,
    FFTW,
    FillArrays,
    ForwardDiff,
    LinearAlgebra,
    LowLevelParticleFilters,
    MonteCarloMeasurements,
    Optim,
    Parameters,
    Random,
    RecipesBase,
    Statistics,
    StatsBase,
    TotalLeastSquares
import DSP
using DSP: filt, filtfilt, impz, Highpass, Lowpass, Bandpass, Bandstop, Butterworth, digitalfilter, FilterType, FilterCoefficients, hamming, hanning, gaussian, xcorr, resample
import StatsBase: predict
import MatrixEquations
import Optim: minimizer, Options
import ControlSystemsBase: ninputs, noutputs, nstates
import StatsBase.residuals
import MatrixEquations
import LowLevelParticleFilters: simulate

export iddata,
    noutputs,
    ninputs,
    nstates,
    input,
    output,
    sampletime,
    hasinput,
    apply_fun,
    resample,
    detrend,
    ramp_in,
    ramp_out,
    timevec,
    freqvec,
    InputOutputData,
    InputOutputFreqData

export AbstractPredictionStateSpace, PredictionStateSpace, N4SIDStateSpace,
    pem, newpem, prediction_error, predict, simulate, noise_model, estimate_x0
export n4sid, subspaceid, era, okid, find_similarity_transform, schur_stab
export getARXregressor,
    getARregressor,
    find_na,
    arx,
    ar,
    arxar,
    residuals,
    arma,
    arma_ssa,
    armax,
    impulseest,
    tls,
    wtls_estimator,
    plr,
    estimate_residuals
export FRD, tfest, coherence, coherenceplot, simplot, simplot!, predplot, predplot!, Hz, rad

export model_spectrum

export KalmanFilter

export weighted_estimator, Bandpass, Bandstop, Lowpass, Highpass, prefilter

export kautz, laguerre, laguerre_oo, laguerre_id, adhocbasis, sum_basis, basis_responses, filter_bank, basislength, ωζ2complex, add_poles, minimum_phase

export autocorplot, crosscorplot, modelfit

include("utils.jl")
include("types.jl")
include("frd.jl")
include("pem.jl")
include("arx.jl")
include("subspace.jl")
include("subspace2.jl")
include("spectrogram.jl")
include("frequency_weights.jl")
include("basis_functions.jl")
include("plotting.jl")

"""
    predict(sys, d::AbstractIdData, args...)
    predict(sys, y, u, x0 = nothing)

See also [`predplot`](@ref)
"""
predict(sys, d::AbstractIdData, args...; kwargs...) =
    hasinput(sys) ? predict(sys, output(d), input(d), args...; kwargs...) :
    predict(sys, output(d), args...; kwargs...)


function predict(sys, y, u, x0 = nothing; h=1)
    h == Inf && return simulate(sys, u, x0)
    h == 1 || throw(ArgumentError("prediction horizon h > 1 not supported for sys"))
    x0 = get_x0(x0, sys, iddata(y, u, sys.Ts))
    model = SysFilter(sys, copy(x0))
    yh = [model(yt, ut) for (yt, ut) in observations(y, u)]
    oftype(y, yh)
end
predict(sys::ControlSystemsBase.TransferFunction, args...; kwargs...) = predict(ss(sys), args...; kwargs...)

get_x0(::Nothing, sys, d::AbstractIdData) = estimate_x0(sys, d)
get_x0(::Nothing, sys, u::AbstractArray) = zeros(sys.nx)
get_x0(::AbstractArray, sys, u::AbstractArray) = zeros(sys.nx)
get_x0(x0::AbstractArray, args...) = x0
function get_x0(s::Symbol, sys, d::AbstractIdData)
    if s ∈ (:zero, :zeros)
        return zeros(sys.nx)
    elseif s === :estimate
        return estimate_x0(sys, d)
    else
        throw(ArgumentError("Unknown option $s. Provide x0 = {:zero, :estimate}"))
    end
end

"""
    slowest_time_constant(sys::AbstractStateSpace{<:Discrete})

Return the slowest time constant of `sys` rounded to the nearest integer samples.
"""
function slowest_time_constant(sys::AbstractStateSpace{<:Discrete})
    Wn, zeta, ps = damp(sys)
    t_const = maximum(1 ./ (Wn.*zeta))
    round(Int, t_const / sys.Ts)
end

slowest_time_constant(sys::TransferFunction{<:Discrete}) = slowest_time_constant(ss(sys))

"""
    estimate_x0(sys, d, n = min(length(d), 3 * slowest_time_constant(sys)); fixed = fill(NaN, sys.nx)

Estimate the initial state of the system 

# Arguments:
- `d`: [`iddata`](@ref)
- `n`: Number of samples to use.
- `fixed`: If a vector of the same length as `x0` is provided, finite values indicate fixed values that are not to be estimated, while nonfinite values are free.

# Example
```julia
sys   = ssrand(2,3,4, Ts=1)
x0    = randn(sys.nx)
u     = randn(sys.nu, 100)
y,t,x = lsim(sys, u; x0)
d     = iddata(y, u, 1)
x0h   = estimate_x0(sys, d, 8, fixed=[Inf, x0[2], Inf, Inf])
x0h[2] == x0[2] # Should be exact equality
norm(x0-x0h)    # Should be small
```
"""
function estimate_x0(sys::AbstractStateSpace, d, n = min(length(d), 3slowest_time_constant(sys)); fixed = fill(NaN, sys.nx))
    d.ny == sys.ny || throw(ArgumentError("Number of outputs of system and data do not match"))
    d.nu == sys.nu || throw(ArgumentError("Number of inputs of system and data do not match"))
    T = ControlSystemsBase.numeric_type(sys)
    y = output(d)
    u = input(d)
    nx,p,N = sys.nx, sys.ny, length(d)
    size(y,2) >= nx || throw(ArgumentError("y should be at least length sys.nx"))

    if sys isa AbstractPredictionStateSpace && !iszero(sys.K)
        ε, _ = lsim(prediction_error_filter(sys), predictiondata(d))
        y = y - ε # remove influence of innovations
    end 

    uresp, _ = lsim(sys, u)
    y = y - uresp # remove influence of u
    # Construct a basis from the response of the system from each component of the initial state
    φx0 = zeros(T, p, N, nx)
    for j in 1:nx
        x0 = zeros(nx); x0[j] = 1
        y0 = lsim(sys, 0*u; x0)[1]
        φx0[:, :, j] = y0 
    end
    φ = reshape(φx0, p*N, :)
    A = φ[1:n*p,:] 
    b = vec(y)[1:n*p]
    if all(!isfinite, fixed)
        return A \ b
    else
        fixvec = isfinite.(fixed)
        b .-= A[:, fixvec] * fixed[fixvec] # Move fixed values over to rhs
        x0free = A[:, .!fixvec] \ b # Solve with remaining free variables
        x0 = zeros(T, sys.nx)
        x0[fixvec] .= fixed[fixvec]
        x0[.!fixvec] .= x0free
        return x0
    end
end

estimate_x0(G::TransferFunction, args...; kwargs...) = estimate_x0(ss(G), args...; kwargs...)

"""
	yh = predict(ar::TransferFunction, y)

Predict AR model
"""
function predict(G::ControlSystemsBase.TransferFunction, y; h=1)
    h == 1 || throw(ArgumentError("prediction horizon h > 1 not supported for sys"))
    _, a, _, _ = params(G)
    yr, A = getARregressor(vec(output(y)), length(a))
    yh = A * a
    oftype(output(y), yh)
end

"""
    simulate(sys, u, x0 = nothing)
    simulate(sys, d, x0 = nothing)

See also [`simplot`](@ref), [`predict`](@ref)
"""
function simulate(sys, u, x0 = nothing)
    x0 = get_x0(x0, sys, u)
    u = input(u)
    model = SysFilter(sys, copy(x0))
    yh = map(observations(u, u)) do (ut, _)
        model(ut)
    end
    oftype(u, yh)
end
simulate(sys::ControlSystemsBase.TransferFunction, args...) = simulate(ss(sys), args...)

function ControlSystemsBase.lsim(sys::AbstractStateSpace, d::AbstractIdData; x0 = nothing)
    d.nu == sys.nu || throw(ArgumentError("Number of inputs of system and data do not match"))
    
    if d.ny == sys.ny
        x0 = get_x0(x0, sys, d)
    else
        x0 = get_x0(x0, sys, d.u)
    end
    lsim(sys, input(d); x0)
end


"""
    noise_model(sys::AbstractPredictionStateSpace)

Return a model of the noise driving the system, `v`, in
```math
x' = Ax + Bu + Kv\\\\
y = Cx + Du + v
```

The model neglects u and is given by
```math
x' = Ax + Kv\\\\
y = Cx + v
```
Also called the "innovation form". This function calls `ControlSystemsBase.innovation_form`.
"""
noise_model(sys::AbstractPredictionStateSpace) = innovation_form(sys)

function ControlSystemsBase.innovation_form(sys::AbstractPredictionStateSpace)
    innovation_form(sys, sys.K)
end


"""
    observer_predictor(sys::N4SIDStateSpace; h=1)

Return the predictor system
```math
x' = (A - KC)x + (B-KD)u + Ky \\\\
y  = Cx + Du
```
with the input equation [B-KD K] * [u; y]

`h ≥ 1` is the prediction horizon.

See also `noise_model` and `prediction_error_filter`.
"""
function ControlSystemsBase.observer_predictor(sys::AbstractPredictionStateSpace; kwargs...)
    K = sys.K
    ControlSystemsBase.observer_predictor(sys, K; kwargs...)
end

"""
    predictiondata(d::AbstractIdData)

Add the output `y` to the input `u_new = [u; y]`
"""
function predictiondata(d::AbstractIdData)
    y,u = output(d), input(d)
    iddata(y, [u; y], d.Ts)
end

"""
    prediction_error_filter(sys::AbstractPredictionStateSpace; h=1)
    prediction_error_filter(sys::AbstractStateSpace, R1, R2; h=1)

Return a filter that takes `[u; y]` as input and outputs the prediction error `e = y - ŷ`. See also `innovation_form` and `noise_model`.
`h ≥ 1` is the prediction horizon. See function [`predictiondata`](@ref) to generate an `iddata` that has `[u; y]` as inputs.
"""
function prediction_error_filter(sys::AbstractStateSpace, args...; kwargs...)
    G = ControlSystemsBase.observer_predictor(sys, args...; kwargs...)
    ss([zeros(sys.ny, sys.nu) I(sys.ny)], sys.Ts) - G
end

Base.@deprecate prediction_error(sys::AbstractStateSpace, args...; kwargs...) prediction_error_filter(sys::AbstractStateSpace, args...; kwargs...)

"""
    e = prediction_error(sys::AbstractStateSpace, d::AbstractIdData, args...; kwargs...)

Return the prediction errors `d.y - predict(sys, d, ...)
"""
function prediction_error(sys::AbstractStateSpace, d::AbstractIdData, args...; kwargs...)
    yh = predict(sys, d, args...; kwargs...)
    d.y .- yh
end

"""
    observer_controller(sys::AbstractPredictionStateSpace, L)

Returns the measurement-feedback controller that takes in `y` and forms the control signal `u = -Lx̂`. See also `ff_controller`. 
"""
function ControlSystemsBase.observer_controller(sys::AbstractPredictionStateSpace, L)
    K = sys.K
    ControlSystemsBase.observer_controller(sys.sys, L, K)
end

"""
    ff_controller(sys::AbstractPredictionStateSpace, L, Lr = static_gain_compensation(sys, L))

Returns the reference controller that takes in `xᵣ` and forms the control signal `u = Lxᵣ`. See also `observer_controller`
"""
function ff_controller(sys::AbstractPredictionStateSpace, L, Lr = static_gain_compensation(sys, L))
    Ae,Be,Ce,De = ssdata(sys)
    K = sys.K
    Ac = Ae - Be*L - K*Ce + K*De*L # 8.26b
    Bc = Be * Lr
    Cc = L
    Dc = 0
    return 1 - ss(Ac, Bc, Cc, Dc, sys.timeevol)
end

function ControlSystemsBase.c2d(sys::AbstractPredictionStateSpace{ControlSystemsBase.Continuous}, Ts::Real; kwargs...)
    sys.S === nothing || iszero(sys.S) || @warn "c2d does not handle a non-zero cross covariance S, S will be set to zero"
    nx, nu, ny = sys.nx, sys.nu, sys.ny
    sysd = c2d(sys.sys, Ts)
    Qd, Rd = c2d(sysd, sys.Q, sys.R)
    M = exp([sys.A.*Ts  sys.K.*Ts;
            zeros(ny, nx+ny)])
    Kd = M[1:nx, nx+1:nx+ny]
    if eltype(sys.A) <: Real
        Kd = real.(Kd)
    end
    PredictionStateSpace(sysd, Kd, Qd, Rd) # modifying R is required to get kalman(sys,Q,R) ≈ K
end


function ControlSystemsBase.d2c(sys::AbstractPredictionStateSpace{<:ControlSystemsBase.Discrete})
    sys.S === nothing || iszero(sys.S) || @warn "d2c does not handle a non-zero cross covariance S, S will be set to zero"
    nx, nu, ny = sys.nx, sys.nu, sys.ny
    Qc = d2c(sys, sys.Q)
    M = log([sys.A  sys.K;
            zeros(ny, nx) I])./sys.Ts
    # Ac = M[1:nx, 1:nx]
    Kc = M[1:nx, nx+1:nx+ny]
    if eltype(sys.A) <: Real
        Kc = real.(Kc)
    end
    PredictionStateSpace(d2c(sys.sys), Kc, Qc, sys.R*sys.Ts) # modifying R is required to get kalman(sys,Q,R) ≈ K
end

"""
    resample(sys::AbstractStateSpace{<:Discrete}, newh::Real)

Change sample-time of sys to `newh`.
"""
function DSP.resample(sys::AbstractStateSpace{<:Discrete}, newh::Real)
    sys.Ts == newh && return sys
    c2d(d2c(sys), newh)
end

"""
    DSP.resample(sys::AbstractStateSpace{<:Discrete}, Qd::AbstractMatrix, newh::Real)

Change sample time of covariance matrix `Qd` beloning to `sys` to `newh`.
This function does not handle the measurement covariance, how to do this depends on context. If the faster sampled signal has the same measurement noise, no change should be made. If the slower sampled signal was downsampled with filtering, the measurement covariance should be increased if the system is changed to a faster sample rate. To maintain the frequency response of the system, the measurement covariance should be modified accordinly.

# Arguments:
- `sys`: A discrete-time system that has dynamics noise covariance matric `Qd`.
- `Qd`: Covariance matrix of dynamics noise.
- `newh`: The new sample time.
"""
function DSP.resample(sys::AbstractStateSpace{<:Discrete}, Qd::AbstractMatrix, newh::Real)
    sys.Ts == newh && return Qd
    sys2 = resample(sys, newh)
    Qc = d2c(sys, Qd)
    c2d(sys2, Qc)
end

end # module
