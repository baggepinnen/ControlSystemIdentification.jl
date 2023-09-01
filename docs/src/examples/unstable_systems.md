# Identification of unstable systems

Unstable systems present several interesting challenges for system identification.
1. The most obvious problem is that it becomes hard to collect experimental data for an unstable system. Imagine trying to estimate a model for a quadrotor, without an already working stabilizing controller, the quadrotor will fall to the ground and crash immediately.
2. With a stabilizing controller, the data collected from the system is comprised of *closed-loop data*, with all the implications that this has on the identification process. See the tutorial [Closed-loop identification](@ref) for more details on this.
3. The final problem is more subtle, it's even impossible to properly simulate the system in open loop! To see this, consider the example below.



## Simulating an unstable system

In this little example, we form the very simple, unstable linear system
```math
\dot{x} = x + u + d
```
where ``u`` is the control input and ``d = \sin(t)`` is a disturbance. We first simulate this system with a stabilizing feedback ``u = -Lx`` and save the data. We then simulate *the same system* again with the saved data as input, but a *slightly* different initial condition. Intuitively, the two simulations should look the same, it's the same system and the same input data, the only difference is the tiny change to the initial condition. 
```@example unstable_systems
using ControlSystemsBase, Plots, LinearAlgebra
sys = tf(1.0, [1, -1]) |> ss # One-state unstable linear system
sys = c2d(sys, 0.01) # Discretize

Q = R = I
L = lqr(sys, Q, R) # Design stabilizing controller

Ts = 0.01 # Sample time
t = 0:Ts:15 # Time vector
inp(x, t) = -L*x .+ sin(t) # Input function is linear feedback + sin(t)

res = lsim(sys, inp, t)
res2 = lsim(sys, res.u, t, x0=[1e-6]) # Simulate again, with the same input data but 1e-6 different initial condition
plot(res)
plot!(res2)
```

The result indicates that the second simulation looks identical in the beginning, but then diverges exponentially. To understand why, we consider the difference between the two simulations in more detail. The first simulation simulated the closed-loop system
```math
\dot{x} = x - Lx + \sin(t) = (1-L)x + \sin(t) 
```
This system is exponentially stable as long as ``L`` is larger than 1. If we collect experimental data for an unstable system under stabilizing feedback, we are actually collecting data from a stable system.

In the second simulation, we simulate the original open-loop system
```math
\dot{x} = x + u + \sin(t)
```
where the control input ``u`` just happened to be collected from a stabilizing controller. So why didn't this second simulation turn out stable? To understand this, consider what happens when we perturb the initial condition ``x_0`` with a tiny perturbation ``x_\delta = x + \delta_x``. If we look at the difference ``\tilde x = x - x_\delta`` between a perturbed and an unperturbed simulation, we get
```math
\dot{\tilde x} = \dot{x} - \dot{x_\delta} = x + u + \sin(t) - (x_\delta + u + \sin(t)) =  (x - x_\delta) = \tilde x
```
This is an exponentially unstable system, and unless ``u`` depends on ``\tilde x``, the difference will grow exponentially. 

We can now understand why the second simulation diverges. The input ``u`` was collected from a stabilizing controller acting on ``x``, but a small deviation in the state occuring after the data has been collected is not corrected by the controller, and the simulation rapidly diverges.

What can we do to improve upon this situation? For pure simulation, there is not much we can do. However, when we are simulating a system for the purpose of estimating its parameters, we have a powerful technique available: The solution is to introduce measurement feedback also in the simulation. Put in other words, instead of simulating the unstable system, we perform *state estimation* on the unstable system using the available measurement data. A state estimator, such as a Kalman filter, has internal measurement feedback that corrects for deviations between the state of the system and the measurement data. For a steady-state Kalman filter in discrete time, this correction looks like
```math
\hat x_t = \hat Ax_{t-1} + K(y_t - \hat y_t) = \hat Ax_{t-1} + KC(x_t - \hat x_t)
```
i.e., we perform a linear correction based on the difference between the predicted output ``\hat y`` and the measured output ``y``. The form of this correction should be very familiar, compare it to the equation of a closed-loop control system with reference ``r``:
```math
\dot x = Ax + Bu, \quad \left[u = L(r - x) \right], \quad \dot x = Ax + BL(r - x) 
```
As long as ``K`` is chosen so as to make the matrix ``A-KC`` exponentially stable, the state estimator will converge to the true state estimate.

## State estimation for system identification
To make use of a state estimator to estimate the parameters of a dynamical system, we could essentially run a state estimator such as a Kalman filter forward in time, compare the estimated states with the measured data, and compute the gradient of the error with respect to the parameters. This is indeed what is done underneath the hood in the Prediction-Error method (PEM) implemented in the function [`newpem`](@ref). This function has a keyword argument `focus` which defaults to `focus = :prediction`, but which can also be chosen as `focus = :simulation` in order to turn the measurement feedback off. However, for unstable systems, the measurement feedback is key to the success of the identification, and any simulation-based algorithm that does not incorporate measurement feedback is prone to divergence.

The PEM algorithm is used to identify a model for the unstable ball-and-beam system in the tutorial [Ball and beam](@ref). Towards the end of this tutorial, the performance of the estimated models are compared using progressively longer prediction horizons, but already with a horizon of 20 steps into the future, the prediction performance has degraded significantly due to the instability.

The prediction-error method has a number of other attractive properties, something that we will explore below.


## Properties of the Prediction-Error Method

Fundamentally, PEM changes the problem from minimizing a loss based on the simulation performance, to minimizing a loss based on shorter-term predictions.[^Ljung][^Larsson] There are several benefits of doing so, and this example will highlight two:

  - The loss is often easier to optimize.
  - In addition to an accurate simulator, you also obtain a prediction for the system.
  - With PEM, it's possible to estimate *disturbance models*.

The last point will not be illustrated in this tutorial, but we will briefly expand upon it here. Gaussian, zero-mean measurement noise is usually not very hard to handle. Disturbances that affect the state of the system may, however, cause all sorts of havoc on the estimate. Consider wind affecting an aircraft, deriving a statistical and dynamical model of the wind may be doable, but unless you measure the exact wind affecting the aircraft, making use of the model during parameter estimation is impossible. The wind is an *unmeasured load disturbance* that affects the state of the system through its own dynamics model. Using the techniques illustrated in this tutorial, it's possible to estimate the influence of the wind during the experiment that generated the data and reduce or eliminate the bias it otherwise causes in the parameter estimates.

We will start by illustrating a common problem with simulation-error minimization. Imagine a pendulum with unknown length that is to be estimated. A small error in the pendulum length causes the frequency of oscillation to change. Over sufficiently large horizon, two sinusoidal signals with different frequencies become close to orthogonal to each other. If some form of squared-error loss is used, the loss landscape will be horribly non-convex in this case, indeed, we will illustrate exactly this below.

Another case that poses a problem for simulation-error estimation is when the system is unstable or chaotic. A small error in either the initial condition or the parameters may cause the simulation error to diverge and its gradient to become meaningless.

In both of these examples, we may make use of measurements we have of the evolution of the system to prevent the simulation error from diverging. For instance, if we have measured the angle of the pendulum, we can make use of this measurement to adjust the angle during the simulation to make sure it stays close to the measured angle. Instead of performing a pure simulation, we instead say that we *predict* the state a while forward in time, given all the measurements until the current time point. By minimizing this prediction rather than the pure simulation, we can often prevent the model error from diverging even though we have a poor initial guess.

We start by defining a model of the pendulum. The model takes a parameter $p = L$ corresponding to the length of the pendulum.

```@example PEM
using Plots, Statistics, DataInterpolations, LowLevelParticleFilters

Ts = 0.01 # Sample time
tsteps = range(0, stop=20, step=Ts)

x0 = [0.0, 3.0] # Initial angle and angular velocity

function pendulum(x, u, p, t) # Pendulum dynamics
    g = 9.82 # Gravitational constant
    L = p isa Number ? p : p[1] # Length of the pendulum
    gL = g / L
    θ = x[1]
    dθ = x[2]
    [dθ
     -gL * sin(θ)]
end
nothing # hide
```

We assume that the true length of the pendulum is $L = 1$, and generate some data from this system.

```@example PEM
using LowLevelParticleFilters: rk4, rollout

discrete_pendulum = rk4(pendulum, Ts) # Discretize the continuous-time dynamics using RK4

function simulate(fun, p)
    x = rollout(fun, x0, tsteps, p; Ts)[1:end-1]
    y = first.(x) # This is the data we have available for parameter estimation (angle measurement)
    x, y
end

x, y = simulate(discrete_pendulum, 1.0) # Simulate with L = 1.0
plot(tsteps, y, title = "Pendulum simulation", label = "angle")
```

We also define functions that simulate the system and calculate the loss, given a parameter `p` corresponding to the length.

```@example PEM
function simloss(p)
    x,yh = simulate(discrete_pendulum, p)
    yh .= abs2.(y .- yh)
    return mean(yh)
end
nothing # hide
```

We now look at the loss landscape as a function of the pendulum length:

```@example PEM
Ls = 0.01:0.01:2
simlosses = simloss.(Ls)
fig_loss = plot(Ls, simlosses,
    title  = "Loss landscape",
    xlabel = "Pendulum length",
    ylabel = "MSE loss",
    lab    = "Simulation loss"
)
```

This figure is interesting, the loss is of course 0 for the true value $L=1$, but for values $L < 1$, the overall slope actually points in the wrong direction! Moreover, the loss is oscillatory, indicating that this is a terrible function to optimize, and that we would need a very good initial guess for a local search to converge to the true value. Note, this example is chosen to be one-dimensional in order to allow these kinds of visualizations, and one-dimensional problems are typically not hard to solve, but the reasoning extends to higher-dimensional and harder problems.

We will now move on to defining a *predictor* model. Our predictor will be very simple, each time step, we will calculate the error $e$ between the simulated angle $\theta$ and the measured angle $y$. A part of this error will be used to correct the state of the pendulum. The correction we use is linear and looks like $Ke = K(y - \theta)$. We have formed what is commonly referred to as a (linear) *observer*. The [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) is a particular kind of linear observer, where $K$ is calculated based on a statistical model of the disturbances that act on the system. We will stay with a simple, fixed-gain observer here for simplicity.

To feed the sampled data into the continuous-time simulation, we make use of an interpolator. We also define new functions, `predictor` that contains the pendulum dynamics with the observer correction, a `prediction` function that performs the rollout (we're not using the word simulation to not confuse with the setting above) and a loss function.

```@example PEM
y_int = LinearInterpolation(y, tsteps)

function predictor(x, u, p, t)
    g = 9.82
    L, K, y = p # pendulum length, observer gain and measurements
    gL = g / L
    θ = x[1]
    dθ = x[2]
    yt = y(t)
    e = yt - θ
    [dθ + K * e
    -gL * sin(θ)]
end

discrete_predictor = rk4(predictor, Ts)

function predloss(p)
    p_full = (p..., y_int)
    x, yh = simulate(discrete_predictor, p_full)
    yh .= abs2.(y .- yh)
    return mean(yh)
end

predlosses = map(Ls) do L
    K = 1.0 # Observer feedback gain
    p = (L, K)
    predloss(p)
end

plot!(Ls, predlosses, lab = "Prediction loss")
```

Once gain, we look at the loss as a function of the parameter, and this time it looks a lot better. The loss is not convex, but the gradient points in the right direction over a much larger interval. Here, we arbitrarily set the observer gain to $K=1$, when we use PEM for estimation, will typically let the optimizer learn this parameter as well, which is what is happening inside [`newpem`](@ref).


Now, we might ask ourselves why we used a correct on the form $Ke$ and didn't instead set the angle in the simulation *equal* to the measurement. The reason is twofold

1. If our prediction of the angle is 100% based on the measurements, the model parameters do not matter for the prediction, and we thus cannot hope to learn their values.
2. The measurement is usually noisy, and we thus want to *fuse* the predictive power of the model with the information of the measurements. The Kalman filter is an optimal approach to this information fusion under special circumstances (linear model, Gaussian noise).


This example has illustrated basic use of the prediction-error method for parameter estimation. In our example, the measurement we had corresponded directly to one of the states, and coming up with an observer/predictor that worked was not too hard. For more difficult cases, we may opt to use a nonlinear observer, such as an extended Kalman filter (EKF) or design a Kalman filter based on a linearization of the system around some operating point.

References:

[^Ljung]: Ljung, Lennart. "System identification---Theory for the user".
[^Larsson]: Larsson, Roger, et al. "Direct prediction-error identification of unstable nonlinear systems applied to flight test data."
