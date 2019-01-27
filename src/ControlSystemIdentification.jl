module ControlSystemIdentification

using DSP, LinearAlgebra, Statistics, Random, Optim, ControlSystems, FillArrays, Parameters, TotalLeastSquares, RecipesBase, FFTW

export StateSpaceNoise, pem, simulation_errors, prediction_errors, predict, simulate, noise_model
export getARXregressor, find_na, arx, bodeconfidence, tls, wtls_estimator, plr


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
