# Statespace model estimation

This page documents the facilities available for estimating linear statespace models with inputs on the form
```math
\begin{aligned}
x^+ &= Ax + Bu + Ke\\
y &= Cx + Du + e
\end{aligned}
```

This package estimates models in discrete time, but they may be converted to continuous-time models using the function [`d2c`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.d2c) from [ControlSystemsBase.jl](https://github.com/JuliaControl/ControlSystems.jl).

There exist several methods for identification of statespace models, [`subspaceid`](@ref), [`n4sid`](@ref), [`newpem`](@ref) and [`era`](@ref). [`subspaceid`](@ref) is the most comprehensive algorithm for subspace-based identification whereas [`n4sid`](@ref) is an older implementation. [`newpem`](@ref) solves the prediction-error problem using an iterative optimization method (from Optim.jl) and ins generally slightly more accurate but also more computationally expensive. If unsure which method to use, try [`subspaceid`](@ref) first (unless the data comes from closed-loop operation, use [`newpem`](@ref) in this case).

## Subspace-based identification using `n4sid` and `subspaceid`
In this example we will estimate a statespace model using the [`subspaceid`](@ref) method. This function returns an object of type [`N4SIDStateSpace`](@ref) where the model is accessed as `sys.sys`.
```@example ss
using ControlSystemIdentification, ControlSystemsBase, Plots
gr(fmt=:png) # hide
Ts = 0.1
G  = c2d(DemoSystems.resonant(), Ts)
u  = randn(1,1000)
y  = lsim(G,u).y
y .+= 0.01 .* randn.() # add measurement noise
d  = iddata(y,u,Ts)
sys = subspaceid(d, :auto; verbose=false, zeroD=true)

# or use a robust version of svd if y has outliers or missing values
# using TotalLeastSquares
# sys = n4sid(d, :auto; verbose=false, svd=x->rpca(x)[3])
bodeplot([G, sys.sys], lab=["True" "" "subspace" ""])
```
[`N4SIDStateSpace`](@ref) is a subtype of `AbstractPredictionStateSpace`, a statespace object that contains an observer gain matrix `sys.K` (Kalman filter) as well as estimated covariance matrices etc.

Using the function [`n4sid`](@ref) instead, we have
```@example ss
sys2 = n4sid(d, :auto; verbose=false, zeroD=true)
bodeplot!(sys2.sys, lab=["n4sid" ""])
```

[`subspaceid`](@ref) allows you to choose the weighting between `:MOESP, :CVA, :N4SID, :IVM` and is generally preferred over [`n4sid`](@ref).

Both functions allow you to choose which functions are used for least-squares estimates and computing the SVD, allowing e.g., robust estimators for resistance against outliers etc.

### Tuning the model fit
The subspace-based estimation algorithms have a number of parameters that can be tuned if the initial model fit is not satisfactory.
- `focus` determines the focus of the model fit. The default is `:prediction` which minimizes the prediction error. If this choice produces an unstable model for a stable system, or the simmulation performance is poor, `focus = :simulation` may be a better choice.
- There are several horizon parameters that can be tuned. The keyword argument `r` selects the prediction horizon, this has to be greater than the model order, with the default being `nx + 10`. `s1` and `s2` control the past horizons for the output and input, respectively. The default is `s1 = s2 = r`. Past horizons can only be tuned for `subspaceid`.
- *(Advanced)* The method used to compute the svd as well as performing least-squares fitting can be changed using the keywords `svd, Aestimator, Bestimator`.
- `zeroD` allows you to force the estimated ``D`` matrix to be zero (a strictly proper model).
- It is possible to select the weight type `W`, choose between `:MOESP, :CVA, :N4SID, :IVM`. The default is `:MOESP`.


See the docstrings of [`subspaceid`](@ref) and [`n4sid`](@ref) for additional arguments and more details.

## ERA and OKID
The "Eigenvalue realization algorithm" and "Observer Kalman identification" algorithms are available as [`era`](@ref) and [`okid`](@ref). If `era` is called with a data object, `okid` is automatically used internally to produce the Markov parameters to the ERA algorithm.
```@example ss
sys3 = era(d, 2) # era has a lot of parameters that may require tuning
bodeplot!(sys3, lab=["ERA" ""])
```

### Using multiple datasets
ERA/OKID supports the use of multiple datasets to improve the estimation accuracy. Below, we show how to perform this manually
```@example ss
using ControlSystemIdentification, ControlSystemsBase, Plots, Statistics
gr(fmt=:png) # hide
Ts = 0.1
G = c2d(tf(1, [1,1,1]), Ts) # True system

# Create several "experiments"
ds = map(1:5) do i
    u = randn(1, 1000)
    y, t, x = lsim(G, u)
    yn = y + 0.2randn(size(y))
    iddata(yn, u, Ts)
end

Ys = okid.(ds, 2, round(Int, 10/Ts), smooth=true, λ=1)    # Estimate impulse response for each experiment
Y = mean(Ys)                            # Average all impulse responses

imp = impulse(G, 10)
f1 = plot(imp, lab="True", l=5)
plot!(imp.t, vec.(Ys), lab="Individual estimates", title="Impulse-response estimates")
plot!(imp.t, vec(Y), l=(3, :black), lab="Mean")

models = era.(Ys, Ts, 2, 50, 50)    # estimate models based on individual experiments
meanmodel = era(Y, Ts, 2, 50, 50)   # estimate model based on mean impulse response

f2 = bodeplot([G, meanmodel], lab=["True" "" "Combined estimate" ""], l=2)
bodeplot!(models, lab="Individual estimates", c=:black, alpha=0.5, legend=:bottomleft)

plot(f1, f2)
```

The procedure shown above is equivalent to calling [`era`](@ref) directly with a vector of data sets, in which case the averaging of the impulse responses is done internally.
```@example ss
era(ds, 2, 50, 50, round(Int, 10/Ts), p=1, λ=1, smooth=true) # Should be identical to meanmodel above
```




## Prediction-error method (PEM)
!!! note "Note"
    The old function [`pem`](@ref) is "soft deprecated" in favor of [`newpem`](@ref) which is more general and much more performant.

The prediction-error method is a simple but powerful algorithm for identification of discrete-time LTI systems on state-space form:
```math
\begin{aligned}
x' &= Ax + Bu + Ke \\
y  &= Cx + Du + e
\end{aligned}
```
The user can choose to minimize either prediction errors or simulation errors, with arbitrary metrics, i.e., not limited to squared errors.

The result of the identification with [`newpem`](@ref) is a custom type with extra fields for the identified Kalman gain and noise covariance matrices.


## Gray-box identification
For estimation of linear or nonlinear models with fixed structure, see [`ControlSystemIdentification.nonlinear_pem`](@ref).


### Usage example
Below, we generate a system and simulate it forward in time. We then try to estimate a model based on the input and output sequences using the function [`newpem`](@ref).
```@example ss
using ControlSystemIdentification, ControlSystemsBase, Random, LinearAlgebra
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

See the [example notebooks](https://github.com/JuliaControl/ControlExamples.jl/blob/master/identification_statespace.ipynb) for more plots as well as several examples in the example section of this documentation.

### Arguments
The algorithm has several options:
- The optimization is by default started with an initial guess provided by [`subspaceid`](@ref), but this can be overridden by providing an initial guess to [`newpem`](@ref) using the keyword argument `sys0`.
- `focus` determines the focus of the model fit. The default is `:prediction` which minimizes the prediction error. If this choice produces an unstable model for a stable system, or the simmulation performance is poor, `focus = :simulation` may be a better choice.
- A regularizer may be provided using the keyword argument `regularizer`.
- A stable model may be enforced using `stable = true`.
- The ``D`` matrix may be forced to be zero using `zeroD = true`.
- A trade-off between prediction and simulation performance can be achieved by optimizing the ``h``-step prediction error. The default is ``h=1`` which corresponds to the standard prediction error. This can be changed using the keyword argument `h`. A large value of `h` will make the optimization problem computationally expensive to solve.

See the docstring of [`newpem`](@ref) for additional arguments and more details.


### Internals
Internally, [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) is used to optimize the system parameters, using automatic differentiation to calculate gradients (and Hessians where applicable). Optim solver options can be controlled by passing keyword arguments to [`newpem`](@ref), and by passing a manually constructed solver object. The default solver is [`BFGS()`](http://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/)




## Filtering, prediction and simulation

When you **estimate** models, you can sometimes select the "focus" of the estimation, to either focus on `:prediciton` performance or `:simulation` performance. Simulation tends to require accurate low-frequency properties, especially for integrating systems, whereas prediction favors an accurate model for higher frequencies. If there are significant input disturbances affecting the system, or if the system is unstable, prediction focus is generally preferred.

When you **validate** the estimated models, you can simulate them using `lsim` from ControlSystemsBase.jl or using [`simulate`](@ref). You may also convert the model to a `KalmanFilter` from [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl) by calling `KalmanFilter(sys)`, after which you can perform filtering and smoothing etc. with the utilities provided for a `KalmanFilter`.

Furthermore, we have the utility functions below
- [`predict`](@ref)`(sys, d, x0=zeros; h=1)`: Form predictions using estimated `sys`, this essentially runs a stationary Kalman filter. `h` denotes the prediction horizon.
- [`simulate`](@ref)`(sys, u, x0=zeros)`: Simulate the system using input `u`. The noise model and Kalman gain does not have any influence on the simulated output.
- [`observer_predictor`](@ref): Extract the predictor model from the estimated system (`ss(A-KC,[B K],C,D)`).
- [`observer_controller`](@ref)
- [`prediction_error`](@ref)
- [`prediction_error_filter`](@ref)
- [`predictiondata`](@ref)
- [`noise_model`](@ref)

## Code generation
To generate C-code for, e.g., simulating a system, see [SymbolicControlSystems.jl](https://github.com/JuliaControl/SymbolicControlSystems.jl).

## Statespace API

```@index
Pages   = ["ss.md"]
```

```@docs
ControlSystemIdentification.newpem
ControlSystemIdentification.subspaceid
ControlSystemIdentification.n4sid
ControlSystemIdentification.era
ControlSystemIdentification.okid
ControlSystemIdentification.observer_predictor
ControlSystemIdentification.observer_controller
ControlSystemIdentification.prediction_error
ControlSystemIdentification.prediction_error_filter
ControlSystemIdentification.noise_model
```

## Video tutorials
Video tutorials performing statespace estimation are available here:

```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/LEZL7UjS2sY?si=u19Q1iIveG6Per3W" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/4fzElP1xeAw?si=bAhIljvyuTHg32H8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/z8o83UORuqQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```