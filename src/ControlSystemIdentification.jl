module ControlSystemIdentification

using BandedMatrices,
    ComponentArrays,
    ControlSystems,
    DelimitedFiles,
    DSP,
    FFTW,
    FillArrays,
    LinearAlgebra,
    LowLevelParticleFilters,
    MonteCarloMeasurements,
    Optim,
    Parameters,
    Random,
    RecipesBase,
    Roots,
    Statistics,
    StatsBase,
    TotalLeastSquares
import StatsBase: predict
import MatrixEquations
import Optim: minimizer, Options
import ControlSystems: ninputs, noutputs, nstates
import StatsBase.residuals

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
    timevec,
    kalman_decomp
export StateSpaceNoise,
    pem, simulation_errors, prediction_errors, predict, simulate, noise_model, estimate_x0
export n4sid, subspaceid, era, okid
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
    bodeconfidence,
    tls,
    wtls_estimator,
    plr,
    estimate_residuals
export FRD, tfest, coherence, coherenceplot, simplot, simplot!, predplot, predplot!, Hz, rad

export model_spectrum

export KalmanFilter

export weighted_estimator, Bandpass, Bandstop, Lowpass, Highpass

export kautz, laguerre, laguerre_oo, adhocbasis, sum_basis, basis_responses, filter_bank, basislength, ωζ2complex, add_poles, minimum_phase

include("utils.jl")
include("types.jl")
include("pem.jl")
include("frd.jl")
include("arx.jl")
include("subspace.jl")
include("subspace2.jl")
include("spectrogram.jl")
include("frequency_weights.jl")
include("basis_functions.jl")

predict(sys, d::AbstractIdData, args...) =
    hasinput(sys) ? predict(sys, output(d), input(d), args...) :
    predict(sys, output(d), args...)


function predict(sys, y, u, x0 = nothing)
    x0 = get_x0(x0, sys, iddata(y,u,sys.Ts))
    model = SysFilter(sys, copy(x0))
    yh = [model(yt, ut) for (yt, ut) in observations(y, u)]
    oftype(y, yh)
end
predict(sys::ControlSystems.TransferFunction, args...) = predict(ss(sys), args...)

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


function estimate_x0(sys, d, n = min(length(d), 10*sys.nx))
    y = output(d)
    u = input(d)
    nx,p,N = sys.nx, sys.ny, length(d)
    size(y,2) >= nx || throw(ArgumentError("y should be at least length sys.nx"))

    if sys isa AbstractPredictionStateSpace
        A,B,C,D = ssdata(sys)
        K = sys.K
        sys = ss(A-K*C, B - K*D, C, D, 1) # TODO: not sure about K*D
        ε = lsim(ss(A-K*C, K, C, 0, 1), y)[1]
        y = y - ε
    end

    uresp = lsim(sys, u)[1]
    y = y - uresp # remove influence of u
    φx0 = zeros(p, N, nx)
    for j in 1:nx
        x0 = zeros(nx); x0[j] = 1
        y0 = lsim(sys, 0*u; x0)[1]
        φx0[:, :, j] = y0 
    end
    φ = reshape(φx0, p*N, :)
    (φ[1:n*p,:]) \ vec(y)[1:n*p]
end

"""
	yh = predict(ar::TransferFunction, y)

Predict AR model
"""
function predict(G::ControlSystems.TransferFunction, y)
    _, a, _, _ = params(G)
    yr, A = getARregressor(output(y), length(a))
    yh = A * a
    oftype(output(y), yh)
end

function simulate(sys, u, x0 = nothing)
    x0 = get_x0(x0, sys, u)
    u = input(u)
    model = SysFilter(sys, copy(x0))
    yh = map(observations(u, u)) do (ut, _)
        model(ut)
    end
    oftype(u, yh)
end
simulate(sys::ControlSystems.TransferFunction, args...) = simulate(ss(sys), args...)


@userplot Simplot
"""
	simplot(sys, data, x0=nothing; ploty=true)

Plot system simulation and measured output to compare them.
`ploty` determines whether or not to plot the measured signal
"""
simplot
@recipe function simplot(p::Simplot; ploty = true)
    sys, d = p.args[1:2]
    y = oftype(randn(2, 2), output(d))
    x0 = length(p.args) > 3 ? p.args[4] : nothing
    x0 = get_x0(x0, sys, d)
    yh = simulate(sys, d, x0)
    xguide --> "Time [s]"
    yguide --> "Output"
    t = timevec(d)
    err = nrmse(y, yh)
    ploty && @series begin
        label --> "y"
        t, y'
    end
    @series begin
        label --> ["sim fit :$(round(err, digits=2))%" for err in err']
        t, yh'
    end
    nothing
end


@userplot Predplot
"""
	predplot(sys, data, x0=nothing; ploty=true)

Plot system simulation and measured output to compare them.
`ploty` determines whether or not to plot the measured signal
"""
predplot
@recipe function predplot(p::Predplot; ploty = true)
    sys, d = p.args[1:2]
    y = oftype(randn(2, 2), output(d))
    u = oftype(randn(2, 2), input(d))
    x0 = length(p.args) > 3 ? p.args[4] : :estimate
    x0 = get_x0(x0, sys, d)
    yh = predict(sys, y, u, x0)
    xguide --> "Time [s]"
    yguide --> "Output"
    t = timevec(d)
    err = nrmse(y, yh)
    ploty && @series begin
        label --> "y"
        t, y'
    end
    @series begin
        label --> ["pred fit :$(round(err, digits=2))%" for err in err']
        t, yh'
    end
    nothing
end

function ControlSystems.lsim(sys::StateSpaceNoise, u; x0 = nothing)
    x0 = get_x0(x0, sys, u)
    simulate(sys, input(u), x0)
end

function ControlSystems.lsim(sys::AbstractStateSpace, d::AbstractIdData; x0 = nothing)
    d.nu == sys.nu || throw(ArgumentError("Number of inputs of system and data do not match"))
    
    if d.ny == sys.ny
        x0 = get_x0(x0, sys, d)
    else
        x0 = get_x0(x0, sys, d.u)
    end
    lsim(sys, input(d); x0)
end


"""
    noise_model(sys::Union{StateSpaceNoise, N4SIDStateSpace})

Return a model of the noise driving the system, `v`, in
x' = Ax + Bu + Kv
y = Cx + Du + v

The model neglects u and is given by
x' = Ax + Kv
y = Cx + v
"""
function noise_model(sys::Union{StateSpaceNoise,N4SIDStateSpace})
    A,B,C,D = ssdata(sys)
    K = sys.K
    G = ss(A, K, C, zeros(size(D,1), size(K, 2)), sys.Ts)
end


"""
    predictor(sys::N4SIDStateSpace)
    predictor(sys::StateSpaceNoise)

Return the predictor system
x' = (A - KC)x + (B-KD)u + Ke
y  = Cx + Du + e
with the input equation [B K] * [u; y]

See also `noise_model` and `prediction_error`.
"""
function ControlSystems.predictor(sys::Union{StateSpaceNoise,N4SIDStateSpace})
    K = sys.K
    ControlSystems.predictor(sys, K)
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
    prediction_error(sys::Union{StateSpaceNoise, N4SIDStateSpace})

Return a filter that takes `[u; y]` as input and outputs the prediction error `e = y - ŷ`. See also `innovation_form` and `noise_model`.
"""
function prediction_error(sys::Union{StateSpaceNoise,N4SIDStateSpace})
    G = ControlSystems.predictor(sys)
    ss([zeros(sys.ny, sys.nu) I(sys.ny)], sys.Ts) - G
end

"""
    ControlSystems.c2d(sys::AbstractStateSpace{<:ControlSystems.Discrete}, Q::AbstractMatrix)

Sample a continuous-time covariance matrix to fit the provided discrete-time system.

The method used comes from theorem 5 in the reference below.

Ref: Discrete-time Solutions to the Continuous-time
Differential Lyapunov Equation With
Applications to Kalman Filtering
Patrik Axelsson and Fredrik Gustafsson
"""
function ControlSystems.c2d(sys::AbstractStateSpace{<:ControlSystems.Discrete}, Qc::AbstractMatrix, R=nothing)
    Ad  = sys.A
    Ac  = real(log(Ad)./sys.Ts)
    h   = sys.Ts
    C   = Symmetric(Qc - Ad*Qc*Ad')
    Qd  = ControlSystems.MatrixEquations.lyapc(Ac, C)
    # The method below also works, but no need to use quadgk when MatrixEquations is available.
    # function integrand(t)
    #     Ad = exp(t*Ac)
    #     Ad*Qc*Ad'
    # end
    # Qd = quadgk(integrand, 0, h)[1]
    if R === nothing
        return Qd
    else
        Qd, R ./ h
    end
end

"""
    d2c(sys::AbstractStateSpace{<:ControlSystems.Discrete}, Qd::AbstractMatrix)

Resample discrete-time covariance matrix belonging to `sys` to the equivalent continuous-time matrix.

The method used comes from theorem 5 in the reference below.

Ref: Discrete-time Solutions to the Continuous-time
Differential Lyapunov Equation With
Applications to Kalman Filtering
Patrik Axelsson and Fredrik Gustafsson
"""
function ControlSystems.d2c(sys::AbstractStateSpace{<:ControlSystems.Discrete}, Qd::AbstractMatrix)
    Ad = sys.A
    Ac = real(log(Ad)./sys.Ts)
    C = Symmetric(Ac*Qd + Qd*Ac')
    Qc = ControlSystems.MatrixEquations.lyapd(Ad, -C)
    isposdef(Qc) || @error("Calculated covariance matrix not positive definite")
    Qc
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
