module ControlSystemIdentification

using DSP, LinearAlgebra, Statistics, Random, Optim, ControlSystems, FillArrays, Parameters, TotalLeastSquares, RecipesBase, FFTW, Roots, MonteCarloMeasurements, LowLevelParticleFilters, BandedMatrices
import MatrixEquations
import Optim: minimizer, Options
import ControlSystems: ninputs, noutputs, nstates

export iddata, noutputs, ninputs, nstates, input, output, sampletime, hasinput, apply_fun, resample, timevec
export StateSpaceNoise, pem, simulation_errors, prediction_errors, predict, simulate, noise_model
export n4sid, era, okid
export getARXregressor, getARregressor, find_na, arx, ar, arma, arma_ssa, armax, impulseest, bodeconfidence, tls, wtls_estimator, plr, estimate_residuals
export FRD, tfest, coherence, coherenceplot, simplot, simplot!, predplot, predplot!

export model_spectrum

export KalmanFilter

export weighted_estimator, Bandpass, Bandstop, Lowpass, Highpass

include("utils.jl")
include("types.jl")
include("pem.jl")
include("arx.jl")
include("frd.jl")
include("subspace.jl")
include("spectrogram.jl")
include("frequency_weights.jl")

predict(sys, d::AbstractIdData, args...) = hasinput(sys) ? predict(sys, output(d), input(d), args...) : predict(sys, output(d), args...)

function predict(sys, y, u, x0=zeros(sys.nx))
	model = SysFilter(sys, copy(x0))
	yh = [model(yt,ut) for (yt,ut) in observations(y,u)]
	oftype(y,yh)
end
predict(sys::ControlSystems.TransferFunction, args...) = predict(ss(sys), args...)

"""
	yh = predict(ar::TransferFunction, y)

Predict AR model
"""
function predict(G::ControlSystems.TransferFunction, y)
	_,a,_ = params(G)
	yr,A = getARregressor(output(y),length(a))
	yh = A*a
	oftype(output(y),yh)
end

function simulate(sys, u, x0=zeros(sys.nx))
	u = input(u)
	model = SysFilter(sys, copy(x0))
	yh = map(observations(u,u)) do (ut,_)
		model(ut)
	end
	oftype(u,yh)
end
simulate(sys::ControlSystems.TransferFunction, args...) = simulate(ss(sys), args...)


@userplot Simplot
"""
	simplot(sys, data, x0=zeros(sys.nx); ploty=true)

Plot system simulation and measured output to compare them.
`ploty` determines whether or not to plot the measured signal
"""
simplot
@recipe function simplot(p::Simplot; ploty=true)
	sys,d = p.args[1:2]
	y = oftype(randn(2,2), output(d))
	u = oftype(randn(2,2), input(d))
	x0 = length(p.args) > 3 ? p.args[4] : zeros(sys.nx)
	yh = simulate(sys, u, x0)
	xguide --> "Time [s]"
	yguide --> "Output"
	t = timevec(d)
	err = nrmse(y,yh)
	ploty && @series begin
		label --> "y"
		t,y'
	end
	@series begin
		label --> "sim fit :$(round.(err, digits=2))%"
		t,yh'
	end
	nothing
end

@userplot Predplot
"""
	predplot(sys, data, x0=zeros(sys.nx); ploty=true)

Plot system simulation and measured output to compare them.
`ploty` determines whether or not to plot the measured signal
"""
predplot
@recipe function predplot(p::Predplot; ploty=true)
	sys,d = p.args[1:2]
	y = oftype(randn(2,2), output(d))
	u = oftype(randn(2,2), input(d))
	x0 = length(p.args) > 3 ? p.args[4] : zeros(sys.nx)
	yh = predict(sys, y, u, x0)
	xguide --> "Time [s]"
	yguide --> "Output"
	t = timevec(d)
	err = nrmse(y,yh)
	ploty && @series begin
		label --> "y"
		t,y'
	end
	@series begin
		label --> "pred fit :$(round.(err, digits=2))%"
		t,yh'
	end
	nothing
end

function ControlSystems.lsim(sys::StateSpaceNoise, u; x0=zeros(sys.nx))
	simulate(sys, input(u), x0)
end

end # module
