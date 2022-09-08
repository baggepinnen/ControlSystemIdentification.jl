# LTI state-space models

This page documents the facilities available for estimating statespace models on the form
```math
\begin{aligned}
x^+ &= Ax + Bu + Ke\\
y &= Cx + Du + e
\end{aligned}
```

There exist several methods for identification of statespace models, [`subspaceid`](@ref), [`n4sid`](@ref), [`newpem`](@ref) and [`era`](@ref). [`subspaceid`](@ref) is the most comprehensive algorithm for subspace-based identification whereas [`n4sid`](@ref) is an older implementation. [`newpem`](@ref) solves the prediction-error problem using an iterative optimization method (from Optim.jl) and ins generally slightly more accurate but also more computationally expensive. If unsure which method to use, try [`subspaceid`](@ref) first (unless the data comes from closed-loop operation, use [`newpem`](@ref) in this case).

## Subspace-based identification using `n4sid` and `subspaceid`
In this example we will estimate a statespace model using the [`n4sid`](@ref) method. This function returns an object of type [`N4SIDStateSpace`](@ref) where the model is accessed as `sys.sys`.
```@example ss
using ControlSystemIdentification, ControlSystems
Ts = 0.1
G  = c2d(DemoSystems.resonant(), Ts)
u  = randn(1,1000)
y  = lsim(G,u).y
y .+= 0.01 .* randn.() # add measurement noise
d  = iddata(y,u,Ts)
sys = n4sid(d, :auto; verbose=false, zeroD=true)
# or use a robust version of svd if y has outliers or missing values
# using TotalLeastSquares
# sys = n4sid(d, :auto; verbose=false, svd=x->rpca(x)[3])
bodeplot([G, sys.sys], lab=["True" "" "n4sid" ""])
```
[`N4SIDStateSpace`](@ref) is a subtype of `AbstractPredictionStateSpace`, a statespace object that contains an observer gain matrix `sys.K` (Kalman filter) as well as estimated covariance matrices etc.

Using the function [`subspaceid`](@ref) instead, we have
```@example ss
sys2 = subspaceid(d, :auto; verbose=false, zeroD=true)
bodeplot!(sys2.sys, lab=["subspace" ""])
```
[`subspaceid`](@ref) allows you to choose the weighting between `:MOESP, :CVA, :N4SID, :IVM` and is generally preferred over [`n4sid`](@ref).

Both functions allow you to choose which functions are used for least-squares estimates and computing the SVD, allowing e.g., robust estimators for resistance against outliers etc.

## ERA and OKID
The "Eigenvalue realization algorithm" and "Observer Kalman identification" algorithms are available as [`era`](@ref) and [`okid`](@ref). If `era` is called with a data object, `okid` is automatically used internally to produce the Markov parameters to the ERA algorithm.
```@example ss
sys3 = era(d, 2)
bodeplot!(sys3, lab=["ERA" ""])
```



## PEM (Prediction-error method)
!!! note "Note"
    The old function [`pem`](@ref) is "soft deprecated" in favor of [`newpem`](@ref) which is more general and much more performant.

A simple algorithm for identification of discrete-time LTI systems on state-space form:
```math
\begin{aligned}
x' &= Ax + Bu + Ke \\
y  &= Cx + Du + e
\end{aligned}
```
is provided. The user can choose to minimize either prediction errors or simulation errors, with arbitrary metrics, i.e., not limited to squared errors.

The result of the identification with [`newpem`](@ref) is a custom type with extra fields for the identified Kalman gain and noise covariance matrices.

### Usage example
Below, we generate a system and simulate it forward in time. We then try to estimate a model based on the input and output sequences using the function [`newpem`](@ref).
```@example ss
using ControlSystemIdentification, ControlSystemsBase Random, LinearAlgebra
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
y .+= 0.01 .* randn.()          # Add some measurement noise
d   = iddata(y,u,0.1)

sysh,opt = newpem(d, nx, focus=:prediction) # Estimate model

yh = predict(sysh, d) # Predict using estimated model
predplot(sysh, d)     # Plot prediction and true output
```

See the [example notebooks](
https://github.com/JuliaControl/ControlExamples.jl/blob/master/identification_statespace.ipynb) for more plots as well as several examples in the example section of this documentation.


### Internals
Internally, [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) is used to optimize the system parameters, using automatic differentiation to calculate gradients (and Hessians where applicable). Optim solver options can be controlled by passing keyword arguments to [`newpem`](@ref), and by passing a manually constructed solver object. The default solver is [`BFGS()`](http://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/)



## Filtering, prediction and simulation
Models can be simulated using `lsim` from ControlSystemsBase.jl and using [`simulate`](@ref). You may also convert the model to a [`KalmanFilter`](@ref) from [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl) by calling `KalmanFilter(sys)`, after which you can perform filtering and smoothing etc. with the utilities provided for a `KalmanFilter`.

Furthermore, we have the utility functions below
- [`predict`](@ref)`(sys, d, x0=zeros; h=1)`: Form predictions using estimated `sys`, this essentially runs a stationary Kalman filter. `h` denotes the prediction horizon.
- [`simulate`](@ref)`(sys, u, x0=zeros)`: Simulate the system using input `u`. The noise model and Kalman gain does not have any influence on the simulated output.
- [`observer_predictor`](@ref): Extract the predictor model from the estimated system (`ss(A-KC,[B K],C,D)`).
- [`observer_controller`](@ref)
- [`prediction_error`](@ref)
- [`prediction_error_filter`](@ref)
- [`predictiondata`](@ref)
- [`noise_model`](@ref)


```@docs
ControlSystemIdentification.subspaceid
ControlSystemIdentification.n4sid
ControlSystemIdentification.newpem
ControlSystemIdentification.era
ControlSystemIdentification.okid
```