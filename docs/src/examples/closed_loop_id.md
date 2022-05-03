# Closed-loop identification
This example will investigate how different identification algorithms perform on closed-loop data, i.e., when the input to the system is produced by a controller using output feedback.

We will consider a very simple system $G(z) = \dfrac{1}{z - 0.9}$ with colored output noise and various inputs formed by state feedback $u = -Lx + r(t)$, where $r$ will vary between the experiments.

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

The ARX and PEM methods are theoretically unbiased in the presence of output feedback, see [^Ljung, Ch 13], while the subspace-based method is not. (Note: the subspace-based method is used to form the initial guess for the iterative PEM algorithm)

```@example closedloop
using ControlSystems, ControlSystemIdentification, Plots
G = tf(1, [1, -0.9], 1) # True system

function generate_data(u; T)
    res = lsim(G, u, 1:T, x0 = [1])
    d = iddata(res)
    E = c2d(tf(1 / 100, [1, 0.01, 0.1]), 1)
    e = lsim(E, randn(1, T)).y
    d.y .+= e
    d
end

function estimate_and_plot(d, nx=1; title)
    Gh1 = arx(d, 1, 1)

    sys0 = subspaceid(d, nx)
    tf(sys0)

    Gh2, _ = ControlSystemIdentification.newpem(d, nx; sys0)
    tf(Gh2)

    bodeplot(
        [G, Gh1, sys0.sys, Gh2.sys];
        ticks = :default,
        title,
        lab = ["True system" "ARX" "Subspace" "PEM"],
        plotphase = false,
    )
end
```

In the first experiment, we have no reference excitation, with a small amount of data ($T=80$), we get terrible estimates
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


We now consider a simple, periodic excitation $r = \sin(t)$
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

With a more complex excitation (random white-spectrum noise), all methods perform well
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

If the feedback is strong but the excitation is weak, the results are rather poor for all methods, it's thus important to have enough energy in the excitation compared to the feedback path.
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


[^Ljung, Ch 13]: Ljung, Lennart. "System identification---Theory for the user".