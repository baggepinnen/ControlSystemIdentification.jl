# LTI state-space models

There exist several methods for identification of statespace models, [`subspaceid`](@ref), [`n4sid`](@ref) and [`pem`](@ref). [`subspaceid`](@ref) is the most comprehensive algorithm for subspace-based identification whereas `n4sid` is an older implementation. [`pem`](@ref) solves the prediction-error problem using an iterative optimization method (from Optim.jl). If unsure which method to use, try [`subspaceid`](@ref) first.

## Subspace-based identification using n4sid
```@example ss
using ControlSystemIdentification, ControlSystems
Ts = 1
G  = ssrand(1,1,3; Ts, proper=true)
u  = randn(1,1000)
y  = lsim(G,u).y
d  = iddata(y,u,Ts)
sys = n4sid(d, :auto; verbose=false, zeroD=true)
# or use a robust version of svd if y has outliers or missing values
# using TotalLeastSquares
# sys = n4sid(d, :auto; verbose=false, svd=x->rpca(x)[3])
bodeplot([G, sys.sys])
```
Estimate a statespace model using the [`n4sid`](@ref) method. Returns an object of type [`N4SIDResult`](@ref) where the model is accessed as `sys.sys`. The frequency-weighting functionality is borrowing ideas from
*"Frequency Weighted Subspace Based System Identification in the Frequency Domain", Tomas McKelvey 1996*. In particular, we apply the output frequency weight matrix (Fy) as it appears in eqs. (16)-(18).

Using the function [`subspaceid`](@ref) instead, we have
```@example ss
sys2 = subspaceid(d, :auto; verbose=false, zeroD=true)
bodeplot!(sys2.sys)
```
[`subspaceid`](@ref) allows you to choose the weighting between `:MOESP, :CVA, :N4SID, :IVM` and is generally preferred over [`n4sid`](@ref).

Both functions allow you to choose which functions are used for least-squares estimates and computing the SVD.

## ERA and OKID
The "Eigenvalue realization algorithm" and "Observer Kalman identification" algorithms are available as [`era`](@ref) and [`okid`](@ref).
```@example ss
sys3 = era(d, 3)
bodeplot!(sys3)
```



## PEM (Prediction-error method)
A simple algorithm for identification of discrete-time LTI systems on state-space form:
```math
x' = Ax + Bu + Ke
```
```math
y  = Cx + e
```
is provided. The user can choose to minimize either prediction errors or simulation errors, with arbitrary metrics, i.e., not limited to squared errors.

The result of the identification with [`pem`](@ref) is a custom type `StateSpaceNoise <: ControlSystems.LTISystem`, with fields `A,B,K`, representing the dynamics matrix, input matrix and Kalman gain matrix, respectively. The observation matrix `C` is not stored, as this is always given by `[I 0]` (you can still access it through `sys.C` thanks to `getproperty`).

### Usage example
Below, we generate a system and simulate it forward in time. We then try to estimate a model based on the input and output sequences.
```@example ss
using ControlSystemIdentification, ControlSystems, Random, LinearAlgebra
using ControlSystemIdentification: newpem
sys = c2d(tf(1, [1, 0.5, 1]) * tf(1, [1, 1]), 0.1)

Random.seed!(1)
T   = 1000                      # Number of time steps
nx  = 3                         # Number of poles in the true system
nu  = 1                         # Number of inputs
x0  = randn(nx)                 # Initial state
sim(sys,u,x0=x0) = lsim(ss(sys), u, x0=x0).y # Helper function
u   = randn(nu,T)               # Generate random input
y   = sim(sys, u, x0)           # Simulate system
d   = iddata(y,u,0.1)

sysh,opt = newpem(d, nx, focus=:prediction) # Estimate model

yh = predict(sysh, d)      # Predict using estimated model
predplot(sysh, d)   # Plot prediction and true output
```

See the [example notebooks](
https://github.com/JuliaControl/ControlExamples.jl/blob/master/identification_statespace.ipynb) for more plots.


### Internals
Internally, [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) is used to optimize the system parameters, using automatic differentiation to calculate gradients (and Hessians where applicable). Optim solver options can be controlled by passing keyword arguments to `pem`, and by passing a manually constructed solver object. The default solver is [`BFGS()`](http://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/)



## Filtering and simulation
Models can be simulated using `lsim` from ControlSystems.jl and using [`simulate`](@ref). You may also convert the model to a [`KalmanFilter`](@ref) from [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl) by calling `KalmanFilter(sys)`, after which you can perform filtering and smoothing etc. with the utilities provided for a `KalmanFilter`.

Furthermore, we have the utility functions below
- [`predict`](@ref)`(sys, d, x0=zeros)`: Form predictions using estimated `sys`, this essentially runs a stationary Kalman filter.
- [`simulate`](@ref)`(sys, u, x0=zeros)`: Simulate the system using input `u`. The noise model and Kalman gain does not have any influence on the simulated output.
- [`observer_predictor`](@ref): Extract the predictor model from the estimated system (`ss(A-KC,[B K],C,D)`).
- [`observer_controller`](@ref)
- [`prediction_error`](@ref)
- [`predictiondata`](@ref)
- [`noise_model`](@ref)


```@docs
ControlSystemIdentification.subspaceid
ControlSystemIdentification.n4sid
ControlSystemIdentification.newpem
ControlSystemIdentification.pem
ControlSystemIdentification.era
ControlSystemIdentification.okid
```