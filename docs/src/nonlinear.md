```@setup HW
using ControlSystemIdentification
using LeastSquaresOptim
```


# Identification of nonlinear models

This package supports two forms of nonlinear system identification.
- Parameter estimation in a known nonlinear model structure ``x⁺ = f(x, u, p)`` where `p` is a vector of parameters to be estimated.
- Estimation of Hammerstein-Wiener models, i.e., linear systems with static nonlinear functions on the input and/or output.

## Parameter estimation in a known nonlinear model structure
Parameter estimation in differential equations can be performed by forming a one-step ahead predictor of the output, and minimizing the prediction error. This procedure is packaged in the function [`nonlinear_pem`](@ref) which is available as a package extension, available if the user manually installs and loads [LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl).

The procedure to use this function is as follows
1. The dynamics is specified on the form ``x_{k+1} = f(x_k, u_k, p, t)`` or ``ẋ = f(x, u, p, t)`` where ``x`` is the state, ``u`` the input,  `p` is a vector of parameters to be estimated and ``t`` is time.
2. If the dynamics is in continuous time (a differential equation or differential-algebraic equation), use the package [SeeToDee.jl](https://github.com/baggepinnen/SeeToDee.jl) to _discretize_ it. If the dynamics is already in discrete time, skip this step.
3. Define the measurement function ``y = h(x, u, p, t)`` that maps the state and input to the measured output.
4. Specify covariance properties of the dynamics noise and measurement noise, similar to how one would do when building a Kalman filter for a linear system.
5. Perform the estimation using [`nonlinear_pem`](@ref).


Internally, [`nonlinear_pem`](@ref) constructs an Unscented Kalman filter (UKF) from the package [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl) in order to perform state estimation along the provided data trajectory. An optimization problem is then solved in order to find the parameters (and optionally initial condition) that minimizes the prediction errors. This procedure is somewhat different from simply finding the parameters that make a pure simulation of the system match the data, notably, the prediction-error approach can usually handle very poor initial guesses, unstable systems and even chaotic systems. To learn more about the prediction-error method, see the tutorial [Properties of the Prediction-Error Method](@ref).

```@docs
ControlSystemIdentification.nonlinear_pem
```


### Example: Quad tank

This example considers a quadruple tank, where two upper tanks feed liquid into two lower tanks, depicted in the schematics below. The quad-tank process is a well-studied example in many multivariable control courses, this particular instance of the process is borrowed from the Lund University [introductory course on automatic control](https://control.lth.se/education/engineering-program/frtf05-automatic-control-basic-course-for-fipi/).

![process](https://user-images.githubusercontent.com/3797491/166203096-40539c68-5657-4db3-bec6-f893286056e1.png)

The process has a *cross coupling* between the tanks, governed by a parameters $\gamma_i$: The flows from the pumps are
divided according to the two parameters $γ_1 , γ_2 ∈ [0, 1]$. The flow to tank 1
is $γ_1 k_1u_1$ and the flow to tank 4 is $(1 - γ_1 )k_1u_1$. Tanks 2 and 3 behave symmetrically.

The dynamics are given by
```math
\begin{aligned}
\dot{h}_1 &= \dfrac{-a_1}{A_1   \sqrt{2g h_1}} + \dfrac{a_3}{A_1 \sqrt{2g h_3}} +     \dfrac{γ_1 k_1}{A_1}   u_1 \\
\dot{h}_2 &= \dfrac{-a_2}{A_2   \sqrt{2g h_2}} + \dfrac{a_4}{A_2 \sqrt{2g h_4}} +     \dfrac{γ_2 k_2}{A_2}   u_2 \\
\dot{h}_3 &= \dfrac{-a_3}{A_3 \sqrt{2g h_3}}                         + \dfrac{(1-γ_2) k_2}{A_3}   u_2 \\
\dot{h}_4 &= \dfrac{-a_4}{A_4 \sqrt{2g h_4}}                          + \dfrac{(1-γ_1) k_1}{A_4}   u_1
\end{aligned}
```
where $h_i$ are the tank levels and $a_i, A_i$ are the cross-sectional areas of outlets and tanks respectively.

We start by defining the dynamics in continuous time, and discretize them using the integrator [SeeToDee.Rk4](https://github.com/baggepinnen/SeeToDee.jl) with a sample time of ``T_s = 1`s.


```@example HW
using StaticArrays, SeeToDee

function quadtank(h, u, p, t)
    k1, k2, g = p[1], p[2], 9.81
    A1 = A3 = A2 = A4 = p[3]
    a1 = a3 = a2 = a4 = 0.03
    γ1 = γ2 = p[4]

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0

    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end
measurement(x,u,p,t) = SA[x[1], x[2]]

Ts = 1.0
discrete_dynamics = SeeToDee.Rk4(quadtank, Ts, supersample=2)
```
The output from this system is the water level in the first two tanks, i.e., ``y = [x_1, x_2]``.

We also specify the number of state variables, outputs and inputs as well as a vector of "true" parameters, the ones we will try to estimate.
```@example HW
nx = 4
ny = 2
nu = 2
p_true = [1.6, 1.6, 4.9, 0.2]
```

We then simulate some data from the system to use for identification:
```@example HW
using ControlSystemIdentification, ControlSystemsBase
using ControlSystemsBase.DemoSystems: resonant
using LowLevelParticleFilters
using LeastSquaresOptim
using Random, Plots, LinearAlgebra

# Generate some data from the system
Random.seed!(1)
Tperiod = 200
t = 0:Ts:1000
u1 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2)) .+ 0.25)
u2 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2 .+ pi/2)) .+ 0.25)
u  = vcat.(u1,u2)
u = [u; 0 .* u[1:100]]
x0 = [2.5, 1.5, 3.2, 2.8]
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u, p_true)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.5randn(ny) for y in y] # Add some measurement noise
Y = reduce(hcat, y) # Go from vector of vectors to a matrix
U = reduce(hcat, u) # Go from vector of vectors to a matrix
plot(
    plot(reduce(hcat, x)', title="States", lab=["h1" "h2" "h3" "h4"]),
    plot(U', title="Inputs", lab=["u1" "u2"]),
    plot(Y', title="Outputs", lab=["y1" "y2"]),
)
```

We package the experimental data into an [`iddata`](@ref) object as usual. Finally, we specify the covariance matrices for the dynamics noise and measurement noise as well as a guess for the initial condition. Since we can measure the level in the first two tanks, we use the true initial condition for those tanks, but we pretend that we are quite off when guessing the initial condition for the last two tanks. 

Choosing the covariance matrices can be non-trivial, see the blog post [How to tune a Kalman filter](https://juliahub.com/pluto/editor.html?id=ad9ecbf9-bf83-45e7-bbe8-d2e5194f2240) for some background. Here, we pick some value for ``R_1`` that seems reasonable, and pick a deliberately large value for ``R_2``. Choosing a large covariance of the measurement noise will lead to the state estimator trusting the measurements less, which in turns leads to a smaller feedback correction. This will make the algorithm favor a model that is good at simulation, rather than focusing exclusively on one-step prediction.

Finally, we call the function [`nonlinear_pem`](@ref) to perform the estimation.
```@example HW
d = iddata(Y, U, Ts)
x0_guess = [2.5, 1.5, 1, 2] # Guess for the initial condition
p_guess = [1.4, 1.4, 5.1, 0.25] # Initial guess for the parameters

R1 = Diagonal([0.1, 0.1, 0.1, 0.1])
R2 = 100*Diagonal(0.5^2 * ones(ny))

model = ControlSystemIdentification.nonlinear_pem(d, discrete_dynamics, measurement, p_guess, x0_guess, R1, R2, nu)
```

We can then test how the model performs on the data, and compare with the model corresponding to our initial guess
```@example HW
simplot(model, d, layout=2)

x_guess = LowLevelParticleFilters.rollout(discrete_dynamics, x0_guess, u, p_guess)[1:end-1]
y_guess = measurement.(x_guess, u, 0, 0)
plot!(reduce(hcat, y_guess)', lab="Initial guess")
```

We can also perform a residual analysis to see if the model is able to capture the dynamics of the system
```@example HW
residualplot(model, d)
```

since we are using simulated data here, the residuals are white and there's nothing to worry about. In practice, one should always inspect the residuals to see if there are any systematic errors in the model.

Internally, the returned model object contains the estimated parameters, let's see if they are any good
```@example HW
[p_true p_guess model.p]
```
hopefully, the estimated parameters are close to the true ones.

To customize the implementation of the nonlinear prediction-error method, see a lower-level interface being used in the tutorial [in the documentation of LowLevelParticleFilters.jl](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Using-an-optimizer) which also provides the UKF.

## Hammerstein-Wiener estimation



This package provides elementary identification of nonlinear systems on Hammerstein-Wiener form, i.e., systems with a static input nonlinearity and a static output nonlinearity with a linear system in-between, **where the nonlinearities are known**. The only aspect of the nonlinearities that are optionally estimated are parameters. To formalize this, the estimation method [`newpem`](@ref) allows for estimation of a model of the form


```math
\begin{aligned}
x^+ &= Ax + B u_{nl} \\
y &= Cx + D u_{nl}   \\
u_{nl} &= g_i(u, p)  \\
y_{nl} &= g_o(y, p)
\end{aligned}
```

```
   ┌─────┐   ┌─────┐   ┌─────┐
 u │     │uₙₗ│     │ y │     │ yₙₗ
──►│  gᵢ ├──►│  P  ├──►│  gₒ ├─►
   │     │   │     │   │     │
   └─────┘   └─────┘   └─────┘
```

where `g_i` and `g_o` are static, nonlinear functions that may depend on some parameter vector ``p`` which is optimized together with the matrices ``A,B,C,D``. The procedure to estimate such a model is detailed in the docstring for [`newpem`](@ref).

The result of this estimation is the linear system _without_ the nonlinearities applied, those must be handled manually by the user.

The default optimizer BFGS may struggle with problems including nonlinearities. If you do not get good results, try a different optimizer, e.g., `optimizer = Optim.NelderMead()`.


### Example with simulated data:

The example below identifies a model of a resonant system where the sign of the output is unknown, i.e., the output nonlinearity is given by ``y_{nl} = |y|``. To make the example a bit more realistic, we also simulate colored measurement and input noise, `yn` and `un`. For an example with real data, see [Hammerstein-Wiener estimation of nonlinear belt-drive system](@ref).

```@example HW
using ControlSystemIdentification, ControlSystemsBase
using ControlSystemsBase.DemoSystems: resonant
using Random, Plots

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
    sysh, x0h, opt = newpem(d, nx; output_nonlinearity, show_trace=false, focus = :simulation)
    (; sysh, x0h, opt)
end;

(; sysh, x0h, opt) = argmin(r->r.opt.minimum, results) # Find the model with the smallest cost

yh = simulate(sysh, d, x0h)
output_nonlinearity(yh, nothing) # We need to manually apply the output nonlinearity to the prediction
plot(d.t, [abs.(y); u]', lab=["True nonlinear output" "Input"], seriestype = [:line :steps], layout=(2,1), xlabel="Time")
scatter!(d.t, ynn', lab="Measured nonlinear output", sp=1)
plot!(d.t, yh', lab="Simulation", sp=1, l=:dash)
```
