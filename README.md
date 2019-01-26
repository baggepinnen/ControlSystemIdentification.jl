# ControlSystemIdentification

[![Build Status](https://travis-ci.org/baggepinnen/ControlSystemIdentification.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/ControlSystemIdentification.jl)
[![Coverage Status](https://coveralls.io/repos/github/baggepinnen/ControlSystemIdentification.jl/badge.svg?branch=master)](https://coveralls.io/github/baggepinnen/ControlSystemIdentification.jl?branch=master)
[![codecov](https://codecov.io/gh/baggepinnen/ControlSystemIdentification.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/ControlSystemIdentification.jl)




This package implements a simple algorithm for identification of discrete-time LTI systems on state-space form. The user can choose to minimize either prediction errors or simulation errors, with arbitrary metrics, i.e., not limited to squared errors.

The result of the identification is a custom type `StateSpaceNoise <: ControlSystems.LTISystem`, with fields `A,B,K`, representing the dynamics matrix, input matrix and Kalman gain matrix, respectively. The observation matrix `C` is not stored, as this is always given by `[I 0]` (you can still access it through `sys.C` thanks to `getproperty`).

This package does not support estimating models on the form
```math
Ay = B/F u + C/D w
```
or any of its special cases. Since those models are also LTI systems, estimating a state-space model is in some sense equivalent.

# Usage example
Below, we generate a system and simulate it forward in time. We then try to estimate a model based on the input and output sequences.
```julia
using ControlSystemIdentification, ControlSystems, Random, LinearAlgebra

function ⟂(x)
    u,s,v = svd(x)
    u*v
end
function generate_system(nx,ny,nu)
    U,S  = ⟂(randn(nx,nx)), diagm(0=>0.2 .+ 0.5rand(nx))
    A    = S*U
    B   = randn(nx,nu)
    C   = randn(ny,nx)
    sys = ss(A,B,C,0,1)
end

Random.seed!(1)
T   = 1000  # Number of time steps
nx  = 3     # Number of poles in the true system
nu  = 1     # Number of control inputs
ny  = 1     # Number of outputs
x0  = randn(nx) # Initial state
sim(sys,u,x0=x0) = lsim(sys, u', 1:T, x0=x0)[1]' # Helper function
sys = generate_system(nx,nu,ny)
u  = randn(nu,T)        # Generate random input
y  = sim(sys, u, x0)    # Simulate system

sysh,x0h,opt = pem(y, u, nx=nx, focus=:prediction) # Estimate model

yh = predict(sysh, y, u, x0h) # Predict using estimated model
plot([y; yh]', lab=["y" "ŷ"]) # Plot prediction and true output
```

We can also simulate the system with colored noise, necessitating estimating also noise models.
```julia
σu = 0.1 # Noise variances
σy = 0.1

sysn = generate_system(nx,nu,ny) # Noise system
un   = u + sim(sysn, σu*randn(size(u)),0*x0) # Input + load disturbance
y    = sim(sys, un, x0)
yn   = y + sim(sysn, σy*randn(size(u)),0*x0) # Output + measurement noise

# The system now as 3nx poles, nx for the system dynamics, and nx for each noise model
sysh,x0h,opt = pem(yn,un,nx=3nx, focus=:prediction)
yh = predict(sysh, yn, un, x0h)
plot([y; yh]', lab=["y" "ŷ"])
```

We can have a look at the singular values of a balanced system Gramian:
```julia
s = ss(sysh) # Convert to standard state-space type
s2,G = balreal(s) # Form balanced representation (obs. and ctrb. Gramians are the same
diag(G) # Singular values of Gramians

9-element Array{Float64,1}:
 3.5972307807882844    
 0.19777167699663994   
 0.0622528285731599    
 0.004322765397504325  
 0.004270259700592557  
 0.003243449461350837  
 0.003150873301312319  
 0.0005827927965893053
 0.00029732262107216666
```
Note that there are 3 big singular values, corresponding to the system poles, there are also 2×3 smaller singular values, corresponding to the noise dynamics.

The estimated noise model can be extracted by `noise_model(sys)`, we can visualize it with a bodeplot.
```julia
bodeplot(noise_model(sysh), exp10.(range(-3, stop=0, length=200)), title="Estimated noise dynamics")
```


# `pem`
`sys, x0, opt = pem(y, u; nx, kwargs...)`
## Arguments:
- `y`: Measurements, either a matrix with time along dim 2, or a vector of vectors
- `u`: Control signals, same structure as `y`
- `nx`: Number of poles in the estimated system. Thus number should be chosen as number of system poles plus number of poles in noise models for measurement noise and load disturbances.
- `focus`: Either `:prediction` or `:simulation`. If `:simulation` is chosen, a two stage problem is solved with prediction focus first, followed by a refinement for simulation focus.
- `metric`: A Function determining how the size of the residuals is measured, default `abs2`, but any Function such as `abs` or `x -> x'Q*x` could be used.
- `regularizer(p) = 0`: function for regularization of the parameter vector `p`. The structure of `p` is detailed below. L₂ regularization, for instance, can be achieved by `regularizer = p->sum(abs2, p)`
- `solver` Defaults to `Optim.BFGS()`
- `kwargs`: additional keyword arguments are sent to [`Optim.Options`](http://julianlsolvers.github.io/Optim.jl/stable/#user/config/).

### Structure of parameter vector `p`
```julia
A  = size(nx,ny)
B  = size(nx,nu)
K  = size(nx,ny)
x0 = size(nx)
p  = [A[:];B[:];K[:];x0]
```

## Return values
- `sys::StateSpaceNoise`: identified system. Can be converted to `StateSpace` by `convert(StateSpace, sys)` or `ss(sys)`, but this will discard the Kalman gain matrix, see `noise_model`.
- `x0`: Estimated initial state
- `opt`: Optimization problem structure. Contains info of the result of the optimization problem

# Functions
- `pem`: Main estimation function, see above.
- `predict(sys, y, u, x0=zeros)`: Form predictions using estimated `sys`, this essentially runs a stationary Kalman filter.
- `simulate(sys, u, x0=zeros)`: Simulate the system using input `u`. The noise model and Kalman gain does not have any influence on the simulated output.
- `noise_model`: Extract the noise model from the estimated system (`ss(A,K,C,0)`).

# Internals
Internally, [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) is used to optimize the system parameters, using automatic differentiation to calculate gradients (and Hessians where applicable). Optim solver options can be controlled by passing keyword arguments to `pem`, and by passing a manually constructed solver object. The default solver is [`BFGS()`](http://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/)

# Other resources
- For estimation of linear time-varying models (LTV), see [LTVModels.jl](https://github.com/baggepinnen/LTVModels.jl).
- For estimation of nonlinear ARX models (NARX), see [BasisFunctionExpansions.jl](https://github.com/baggepinnen/BasisFunctionExpansions.jl).
- For estimation of linear and nonlinear grey-box models in continuous time, see [DifferentialEquations.jl (parameter estimation)](http://docs.juliadiffeq.org/stable/analysis/parameter_estimation.html)
- Estimation of nonlinear black-box models in continuous time [DiffEqFlux.jl](https://github.com/JuliaDiffEq/DiffEqFlux.jl/) and in discrete time [Flux.jl](https://github.com/FluxML/Flux.jl)
