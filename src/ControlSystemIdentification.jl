module ControlSystemIdentification

using DSP, LinearAlgebra, Statistics, Random, Optim, ControlSystems, FillArrays, Parameters, TotalLeastSquares, RecipesBase, FFTW, Roots, MonteCarloMeasurements
import Optim: minimizer, Options

export StateSpaceNoise, pem, simulation_errors, prediction_errors, predict, simulate, noise_model
export getARXregressor, getARregressor, find_na, arx, ar, bodeconfidence, tls, wtls_estimator, plr
export FRD, tfest, coherence, coherenceplot, simplot, simplot!, predplot, predplot!


include("utils.jl")
include("types.jl")
include("pem.jl")
include("arx.jl")
include("frd.jl")

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
	yr,A = getARregressor(y,length(a))
	yh = A*a
	oftype(y,yh)
end

function simulate(sys, u, x0=zeros(sys.nx))
	model = SysFilter(sys, copy(x0))
	yh = map(observations(u,u)) do (ut,_)
		model(ut)
	end
	oftype(u,yh)
end
simulate(sys::ControlSystems.TransferFunction, args...) = simulate(ss(sys), args...)


@userplot Simplot
"""
simplot(sys, y, u, x0=zeros(sys.nx); ploty=true)
Plot system simulation and measured output to compare them.
`ploty` determines whether or not to plot the measured signal
"""
simplot
@recipe function simplot(p::Simplot; ploty=true)
	sys,y,u = p.args[1:3]
	y = oftype(randn(2,2), y)
	u = oftype(randn(2,2), u)
	x0 = length(p.args) > 3 ? p.args[4] : zeros(sys.nx)
	yh = simulate(sys,u, x0)
	xguide --> "Time [s]"
	yguide --> "Output"
	t = range(0, step=sys.Ts, length=length(y))
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
predplot(sys, y, u, x0=zeros(sys.nx); ploty=true)
Plot system simulation and measured output to compare them.
`ploty` determines whether or not to plot the measured signal
"""
predplot
@recipe function predplot(p::Predplot; ploty=true)
	sys,y,u = p.args[1:3]
	y = oftype(randn(2,2), y)
	u = oftype(randn(2,2), u)
	x0 = length(p.args) > 3 ? p.args[4] : zeros(sys.nx)
	yh = predict(sys,y,u, x0)
	xguide --> "Time [s]"
	yguide --> "Output"
	t = range(0, step=sys.Ts, length=length(y))
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
	simulate(sys, u, x0)
end

end # module
