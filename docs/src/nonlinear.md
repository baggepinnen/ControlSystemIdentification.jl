# Identification of nonlinear models

For nonlinear graybox identification, i.e., identification of models on a known form but with unknown parameters, we suggest to implement a nonlinear version of the prediction-error method by using an Unscented Kalman filter as predictor. A tutorial for this is available [in the documentation of LowLevelParticleFilters.jl](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Using-an-optimizer) which also provides the UKF.


This package provides very elementary identification of nonlinear systems on Hammerstein-Wiener form, i.e., systems with a static input nonlinearity and a static output nonlinearity with a linear system inbetween, **where the nonlinearities are known**. The only aspect of the nonlinearities that are optionally estimated are parameters of the output nonlinearity.

The procedure to estimate such a model is
1. _Manually_ apply the input nonlinearity (if present) to the input signal `u` _before_ estimation, i.e., use the nonlinearly transformed input in the [`iddata`](@ref) object `d`.
2. If the output nonlinearity _is invertible_, apply the inverse to the output signal `y` _before_ estimation similar to above.
3. If the output nonlinearity _is not invertible_, provide the nonlinear output transformation as a function using the keyword argument `output_nonlinearity` to the estimation algorithm [`newpem`](@ref). The `output_nonlinearity` function is expected to operate on the (vector) output signal `y` and modify it _in-place_. Example:
```julia
function output_nonlinearity(y, p)
    y[1] = y[1] + p[1]*y[1]^2  # Note how the incoming vector is modified in-place
    y[2] = abs(y[2])
end
```
Please note, `y = f(y)` does not change `y` in-place, but creates a new vector `y` and assigns it to the variable `y`. This is not what we want here.

The second argument to `output_nonlinearity` is an (optional) vector of parameters that can be optimized. To use this option, pass the keyword argument `nlp` to [`newpem`](@ref) with a vector of initial guesses for the nonlinear parameters.


The result of this estimation is the linear system _without_ the nonlinearities.

The default optimizer BFGS may struggle with problems including nonlinearities, if you do not get good results, try a different optimizer, e.g., `optimizer = Optim.NelderMead()`.


## Example:

The example below identifies a model of a resonant system with a static where the sign of the output is unknown, i.e., the output nonlinearity is given by ``y_{nl} = |y|``. To make the example a bit more realistic, we also simulate colored measurement and input noise, `yn` and `un`.
```@example HW
using ControlSystemIdentification, ControlSystemsBase
using ControlSystemsBase.DemoSystems: resonant
using Random

# Generate some data from the system
Random.seed!(1)
T = 200
sys = c2d(resonant(ω0 = 0.1) * tf(1, [0.1, 1]), 1)# generate_system(nx, nu, ny)
nx = sys.nx
nu = 1
ny = 1
x0 = zeros(nx)
sim(sys, u, x0 = x0) = lsim(sys, u, 1:T, x0 = x0)[1]
sysn = c2d(resonant(ω0 = 1) * tf(1, [0.1, 1]), 1)

σu = 1e-2 # Input noise standard deviation
σy = 1e-3 # Output noise standard deviation

u  = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)

# Nonlinear output transformation
ynn = abs.(yn)
d  = iddata(ynn, un, 1)
output_nonlinearity = (y, p) -> y .= abs.(y)

# Estimate 10 models with different random initialization and pick the best one
# If results are poor, try `optimizer = Optim.NelderMead()` instead
results = map(1:10) do _
    sysh, x0h, opt = newpem(d, nx; output_nonlinearity, show_trace=false)
    (; sysh, x0h, opt)
end;

(; sysh, x0h, opt) = argmin(r->r.opt.minimum, results) # Find the model with the smallest cost

yh = predict(sysh, d, x0h)
output_nonlinearity(yh, nothing)
plot(d.t, [abs.(y); u]', lab=["True nonlinear output" "Input"], seriestype = [:line :steps], layout=(2,1), xlabel="Time")
scatter!(d.t, ynn', lab="Measured nonlinear output", sp=1)
plot!(d.t, yh', lab="Prediction", sp=1, l=:dash)
```