# Closed-loop identification
This example will investigate how different identification algorithms perform on closed-loop data, i.e., when the input to the system is produced by a controller using output feedback.

We will consider a very simple system $G(z) = \dfrac{1}{z - 0.9}$ with colored output noise and various inputs formed by output feedback $u = -Ly + r(t)$, where $r$ will vary between the experiments.

It is well known that in the absence of $r$ and with a simple regulator, identifiability is poor, indeed, if
```math
y_{k+1} = a y_k + b u_k, \quad u_k = L y_k
```
we get the closed-loop system
```math
y_{k+1} = (a + bL)y_k
```
where we can not distinguish $a$ and $b$. The introduction of $r$ resolves this
```math
\begin{aligned}
y_{k+1} &= a y_k + b u_k \\
u_k &= L y_k + r \\
y_{k+1} &= (a + bL)y_k + b r_k
\end{aligned}
```
The very first experiment below will illustrate the problem when there is no excitation through $r$.

We start by defining a model of the true system, a function that simulates some data and adds colored output noise, as well as a function that estimates three different models and plots their frequency responses. We will consider three estimation methods
1. [`arx`](@ref), a prediction-error approach based on a least-squares estimate.
2. A subspace-based method [`subspaceid`](@ref), known to be biased in the presence of output feedback.
3. The prediction-error method (PEM) [`newpem`](@ref)

The ARX and PEM methods are theoretically unbiased in the presence of output feedback, see [^Ljung], while the subspace-based method is not. (Note: the subspace-based method is used to form the initial guess for the iterative PEM algorithm)

```@example closedloop
using ControlSystemsBase, ControlSystemIdentification, Plots
gr(fmt=:png) # hide
G = tf(1, [1, -0.9], 1) # True system

function generate_data(u; T)
    E = c2d(tf(1 / 100, [1, 0.01, 0.1]), 1) # Noise model for colored noise 
    e = lsim(E, randn(1, T)).y              # Noise realization
    function u_noise(x, t)
        y = x .+ e[t] # Add the measured noise to the state to simulate feedback of measurement noise
        u(y, t)
    end
    res = lsim(G, u_noise, 1:T, x0 = [1])
    d = iddata(res)
    d.y .+= e # Add the measurement noise to the output that is stored in the data object
    d
end

function estimate_and_plot(d, nx=1; title, focus=:prediciton)
    Gh1 = arx(d, 1, 1)

    sys0 = subspaceid(d, nx; focus)
    tf(sys0)

    Gh2, _ = ControlSystemIdentification.newpem(d, nx; sys0, focus)
    tf(Gh2)

    figb = bodeplot(
        [G, Gh1, sys0.sys, Gh2.sys];
        ticks = :default,
        title,
        lab = ["True system" "ARX" "Subspace" "PEM"],
        plotphase = false,
    )

    figd = plot(d)
    plot(figb, figd)
end
```

In the first experiment, we have **no reference excitation**, with a small amount of data ($T=80$), we get terrible estimates
```@example closedloop
L = 0.5 # Feedback gain u = -L*x
u = (x, t) -> -L * x
title = "-Lx"
estimate_and_plot(generate_data(u, T=80), title=title*",  T=80")
```

with a larger amount of data $T=8000$, we get equally terrible estimates
```@example closedloop
estimate_and_plot(generate_data(u, T=8000), title=title*",  T=8000")
```

This indicates that we can not hope to estimate a model if the system is driven by noise only.


We now consider a simple, **periodic excitation** $r = \sin(t)$
```@example closedloop
L = 0.5 # Feedback gain u = -L*x
u = (x, t) -> -L * x .+ 5sin(t)
title = "-Lx + 5sin(t)"
estimate_and_plot(generate_data(u, T=80), title=title*",  T=80")
```

In this case, all but the subspace-based method performs quite well
```@example closedloop
estimate_and_plot(generate_data(u, T=8000), title=title*",  T=8000")
```

More data does not help the subspace method.

With **a more complex excitation** (random white-spectrum noise), all methods perform well
```@example closedloop
L = 0.5 # Feedback gain u = -L*x
u = (x, t) -> -L * x .+ 5randn()
title = "-Lx + 5randn()"
estimate_and_plot(generate_data(u, T=80), title=title*",  T=80")
```

and even slightly better with more data.
```@example closedloop
estimate_and_plot(generate_data(u, T=8000), title=title*",  T=8000")
```

If the **feedback is strong but the excitation is weak**, the results are rather poor for all methods, it's thus important to have enough energy in the excitation compared to the feedback path.
```@example closedloop
L = 1 # Feedback gain u = -L*x
u = (x, t) -> -L * x .+ 0.1randn()
title = "-Lx + 0.1randn()"
estimate_and_plot(generate_data(u, T=80), title=title*",  T=80")
```

In this case, we can try to increase the model order of the PEM and subspace-based methods to see if they are able to learn the noise model (which has two poles)
```@example closedloop
estimate_and_plot(generate_data(u, T=8000), 3, title=title*",  T=8000")
```
learning the noise model can sometimes work reasonably well, but requires more data. You may extract the learned noise model using [`noise_model`](@ref).



## Detecting the presence of feedback
It is sometimes possible to detect the presence of feedback in a dataset by looking at the cross-correlation between input and output. For a causal system, there shouldn't be any correlation for negative lags, but feedback literally feeds outputs back to the input, leading to a reverse causality:
```@example closedloop
L = 0.5 # Feedback gain u = -L*x
u = (x, t) -> -L * x .+ randn.()
title = "-Lx + 5sin(t)"
crosscorplot(generate_data(u, T=500), -5:10, m=:circle)
```
Here, the plot clearly has significant correlation for both positive and negative lag, indicating the presence of feedback. The controller used here is a static P-controller, leading to a one-step correlation backwards in time. With a dynamic controller (like a PI controller), the effect would be more significant.

If we remove the feedback, we get
```@example closedloop
L = 0.0 # no feedback
u = (x, t) -> -L * x .+ randn.()
title = "-Lx + 5sin(t)"
crosscorplot(generate_data(u, T=500), -5:10, m=:circle)
```
now, the correlation for negative lags and zero lag is mostly non-significant (below the dashed lines).



[^Ljung]: Ljung, Lennart. "System identification---Theory for the user", Ch 13.