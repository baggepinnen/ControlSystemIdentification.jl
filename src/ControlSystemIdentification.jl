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
import MatrixEquations

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
    ramp_in,
    timevec
export AbstractPredictionStateSpace, PredictionStateSpace, N4SIDStateSpace, StateSpaceNoise,
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

export weighted_estimator, Bandpass, Bandstop, Lowpass, Highpass, prefilter

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
include("plotting.jl")

"""
    predict(sys, d::AbstractIdData, args...)
    predict(sys, y, u, x0 = nothing)

See also [`predplot`](@ref)
"""
predict(sys, d::AbstractIdData, args...) =
    hasinput(sys) ? predict(sys, output(d), input(d), args...) :
    predict(sys, output(d), args...)


function predict(sys, y, u, x0 = nothing)
    x0 = get_x0(x0, sys, iddata(y, u, sys.Ts))
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

function slowest_time_constant(sys::AbstractStateSpace{<:Discrete})
    Wn, zeta, ps = damp(sys)
    t_const = maximum(1 ./ (Wn.*zeta))
    round(Int, t_const / sys.Ts)
end

"""
    estimate_x0(sys, d, n = min(length(d), 3 * slowest_time_constant(sys)))

Estimate the initial state of the system 

# Arguments:
- `d`: [`iddata`](@ref)
- `n`: Number of samples to used.
"""
function estimate_x0(sys, d, n = min(length(d), 3slowest_time_constant(sys)))
    d.ny == sys.ny || throw(ArgumentError("Number of outputs of system and data do not match"))
    d.nu == sys.nu || throw(ArgumentError("Number of inputs of system and data do not match"))
    T = ControlSystems.numeric_type(sys)
    y = output(d)
    u = input(d)
    nx,p,N = sys.nx, sys.ny, length(d)
    size(y,2) >= nx || throw(ArgumentError("y should be at least length sys.nx"))

    if sys isa AbstractPredictionStateSpace
        # A,B,C,D = ssdata(sys)
        # K = sys.K
        # sys = ss(A-K*C, B - K*D, C, D, 1) 
        # ε = lsim(ss(A-K*C, K, C, 0, 1), y)[1]
        ε, _ = lsim(prediction_error(sys), predictiondata(d))
        y = y - ε
    end 

    uresp, _ = lsim(sys, u)
    y = y - uresp # remove influence of u
    φx0 = zeros(T, p, N, nx)
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
    yr, A = getARregressor(time1(output(y)), length(a))
    yh = A * a
    oftype(output(y), yh)
end

"""
    simulate(sys, u, x0 = nothing)
    simulate(sys, d, x0 = nothing)

See also [`simplot`](@ref)
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
simulate(sys::ControlSystems.TransferFunction, args...) = simulate(ss(sys), args...)



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
    noise_model(sys::AbstractPredictionStateSpace)

Return a model of the noise driving the system, `v`, in
x' = Ax + Bu + Kv
y = Cx + Du + v

The model neglects u and is given by
x' = Ax + Kv
y = Cx + v
"""
function noise_model(sys::AbstractPredictionStateSpace)
    A,B,C,D = ssdata(sys)
    K = sys.K
    G = ss(A, K, C, zeros(size(D,1), size(K, 2)), sys.timeevol)
end


"""
    observer_predictor(sys::N4SIDStateSpace)
    observer_predictor(sys::StateSpaceNoise)

Return the predictor system
x' = (A - KC)x + (B-KD)u + Ke
y  = Cx + Du + e
with the input equation [B K] * [u; y]

See also `noise_model` and `prediction_error`.
"""
function ControlSystems.observer_predictor(sys::AbstractPredictionStateSpace)
    K = sys.K
    ControlSystems.observer_predictor(sys, K)
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
    prediction_error(sys::AbstractPredictionStateSpace)
    prediction_error(sys::AbstractStateSpace, R1, R2)

Return a filter that takes `[u; y]` as input and outputs the prediction error `e = y - ŷ`. See also `innovation_form` and `noise_model`.
"""
function prediction_error(sys::AbstractStateSpace, args...)
    G = observer_predictor(sys, args...)
    ss([zeros(sys.ny, sys.nu) I(sys.ny)], sys.Ts) - G
end

"""
    observer_controller(sys::AbstractPredictionStateSpace, L)

Returns the measurement-feedback controller that takes in `y` and forms the control signal `u = -Lx̂`. See also `ff_controller`. 
"""
function ControlSystems.observer_controller(sys::AbstractPredictionStateSpace, L)
    K = sys.K
    observer_controller(sys.sys, L, K)
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
    Qd  = MatrixEquations.lyapc(Ac, C)
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

function ControlSystems.d2c(sys::AbstractPredictionStateSpace{<:ControlSystems.Discrete})
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
    Qc = MatrixEquations.lyapd(Ad, -C)
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
