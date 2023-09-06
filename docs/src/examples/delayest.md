# Delay estimation
A frequent property of control systems is the presence of _delays_, either due to processing time, network latency, or other physical phenomena. Delays that occur internally in the system will in the discrete-time setting add a potentially large number of states to the system, ``\tau / T_s`` state variables are required to represent a delay of ``\tau`` seconds in a discrete-time system with sampling time ``T_s``. A common special case is that the delay occurs at either the input or the output of the system, e.g, due to communication delays. Estimating this delay ahead of the estimation of the model is often beneficial, since it reduces the number of parameters that need to be estimated. Below, we generate a dataset with a large input delay ``\tau`` and have a look at how we can estimate ``\tau``.

## Estimating the delay


```@example DELAY
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystemsBase

τ = 2.0 # Delay in seconds
Ts = 0.1 # Sampling time
P = c2d(tf(1, [1, 0.5, 1])*delay(τ), Ts) # Dynamics is given by a simple second-order system with input delay

u = sin.(0.1 .* (0:Ts:30).^2) # An interesting input signal
res = lsim(P, u')
d = iddata(res)
plot(d)
```

If we inspect the (discrete-time) system, we see that it indeed has a large-dimensional state:
```@example DELAY
P.nx
```

A simple, non-parametric way of figuring out whether or not there's a delay in the system is to look at the cross-correlation between the input and the output, we can do this using the [`crosscorplot`](@ref) function:

```@example DELAY
crosscorplot(d)
```
The plot indicates that there is insignificant (below the dashed lines) correlation between the input and the output for lags smaller than 2 seconds, corresponding exactly to our delay ``\tau``. 

Another method to identify the delay would be to estimate the impulse response of the system:
```@example DELAY
impulseestplot(d, 60; λ=0.5, lab="Estimated", title="Impulse response of delay system")
plot!(impulse(P, 6), lab="True", framestyle=:zerolines)
```
Estimation of impulse responses in the presence of such large delays is numerically challenging, and a regularization of ``λ=0.5`` was required to achieve a reasonable result. If the delay is expected to be large and the dataset is small, it is thus recommended to use the cross-correlation method instead, since we cannot tune the regularization parameter ``λ`` in a practical setting when we don't know what impulse response to expect. However, for short delays and large datasets, the impulse-response method works rather well. Below, we show the estimated impulse response for a much larger dataset just to demonstrate:
```@example DELAY
u2 = sin.(0.01 .* (0:Ts:300).^2) .+ randn.() # An interesting and long input signal with some noise as well
res2 = lsim(P, u2')
d2 = iddata(res2)

impulseestplot(d2, 200; λ=0.0, lab="Estimated", title="Impulse response with large dataset")
plot!(impulse(P, 20), lab="True", framestyle=:zerolines)
```

A third method, arguably less elegant, is to use a model-selection method to figure out the delay. Presumably, models estiamted with an order smaller than the delay will be rather poor, something that should be visible if we try models of many orders. Below, we use the function [`find_nanb`](@ref) that tries to identify the appropriate model order for an [`arx`](@ref) model.

```@example DELAY
find_nanb(d, 2:8, 10:30, xrotation=90, size=(800, 300), margin=5Plots.mm, xtickfont=6)
```
The plot indicates that the Akaike information criterion (AIC) is minimized when `nb` reaches 22, which happens to be the number of delay samples ``\tau / T_s`` + the number of numerator parameters for the system without delay:
```@example DELAY
c2d(tf(1, [1, 0.5, 1]), Ts)
```


## Handling the delay when estimating a model
Once we know that there is a delay present, we need to somehow handle it while estimating a model. The naive way is to simply select a model with high-enough order and call it a day. However, this is prone to numerical problems and generally not recommended. Some methods have a dedicated `inputdelay` keyword that allows you to specify known input delays, after which the method handles it internally. For methods that do not have this option, one can always preprocess the data to remove the delay, we show how to do this next:

```@example DELAY
τ_samples = 20 # Number of samples delay
d2 = iddata(d.y[:, τ_samples+1:end], d.u[:, 1:end-τ_samples], d.Ts)
crosscorplot(d2)
```
The cross-correlation plot now indicates that the delay is gone. Please note, one sample delay is most of the time expected, indeed, it's unrealistic for the output of a physical system to respond at the same time as something happens at the input. Once we have estimated a model with the dataset with the delay removed, we must add the delay back into the estimated model, e.g.,
```@example DELAY
P̂ = subspaceid(d2, focus=:simulation) # Estimate a model without delay
tf(d2c(P̂)) # Does it match the original continuous-time system without delay?
```
The estimated model above should be very close to the system `tf(1, [1, 0.5, 1])` (some small higher-order terms in the numerator are expected). To add the delay to this model, we do
```@example DELAY
P̂τ = P̂*delay(τ, Ts) # Note that this is different from delay(τ, Ts)*P̂ which adds output delay instead of input delay
bodeplot([P, P̂τ], lab=["True system" "" "Estimated system" ""])
```
The estimated model should now match the true system very well, including the large drop in phase for higher frequencies due to the delay.


!!! warning "Internal delays"
    If the system contains internal delays, it might appear in a cross-correlation plot as if the system has an input-output delay, but in this case shifting the data like we did above may be ill-advised. With internal delays, use the number of estimated delay samples as a lower bound on the model order instead. An example of an internal delay is when a control loop is closed around a delayed channel, illustrated below.


## Internal delays

In discrete time, a one-sample delay on either the input or the output appear as a pole in the origin. However, when the delay is internal to the system, it's not as easy to detect. Below, we close the loop around the system ``P`` from above using a PI controller ``C``, and construct a new dataset where the input is the reference to the PI controller rather than the input to the plant. 

```@example DELAY
C = pid(0.1, 0.5; Ts)
L = P*C
G = feedback(L)
plot(nyquistplot(L), plot(step(G, 50)))
```

```@example DELAY
ref = sign.(sin.(0.02 .* (0:Ts:50).^2)) # An interesting reference signal
res = lsim(G, ref')
dG = iddata(res)
plot(plot(dG), crosscorplot(dG))
```

We see that the cross-correlation plot still indicates that it takes over 2 seconds for the output to be affected by the input, but if we look at the model-selection plot, we must include orders up to above 23 to get a good fit:


```@example DELAY
find_nanb(dG, 3:30, 5:30, xrotation=90, size=(800, 300), margin=5Plots.mm, xtickfont=6)
```

We can also inspect the pole-zero maps of the open-loop and closed-loop systems


```@example DELAY
plot(
    pzmap(P, title="Open-loop system"),
    pzmap(G, title="Closed-loop system"),
    plot_title="Pole-zero maps",
    layout=(1, 2),
    ratio=:equal,
    size=(800, 400),
    xlims=(-1.1, 1.1),
    ylims=(-1.1, 1.1),
)
```
the presence of the feedback has moved all the delay poles away from the origin (there are multiple poles on top of each other in the left plot).

In this case, we must thus estimate a model of fairly high order (23) in order to accurately capture the dynamics of the system. If we do this, we can see that the estimated model matches the true system very well (since we didn't add any disturbance):

```@example DELAY
model = subspaceid(dG, G.nx)
simplot(model, dG, zeros(model.nx))
```


In a practical scenario, estimating a model of such high order may be difficult. It might then be worthwhile trying to estimate the lower-order open-loop system ``P`` directly, taking care to properly handle the delay, which then appears on the input rather than internally.

If we use one of the methods that support the `inputdelay` keyword, we can see whether or not we can approximate the closed-loop system with a model that has an input delay only:
```@example DELAY
model2 = arx(dG, 4, 4, inputdelay=τ_samples)
plot(
    simplot(model2, dG),
    pzmap(
        model2,
        ratio=:equal,
        xlims=(-1.1, 1.1),
        ylims=(-1.1, 1.1),
        title="Estimated ARX model with input delay",
    )
)
```
We see that with a model of order 3 (4 free parameters in the denominator) together with a specified input delay of 20 samples, we indeed get a good approximation of the closed-loop system.

If we compare the Bode plots of the true closed-loop system with the two identified models, they match very well.
```@example DELAY
bodeplot([G, model, model2])
```

It may thus be possible to approximate a system with internal delays using a model that has an input delay only.


For completeness, we construct an example where this is not quite possible. The system in the example below can be thought of as an echo chamber, where the input passes through a resonant channel before it reaches the output, and 45% of the output energy is fed back at the input through the same channel (the echo)
    
```@example DELAY
ref = sign.(sin.(0.02 .* (0:Ts:150).^2)) # An interesting reference signal
Pc = feedback(tf(1, [1, 1, 1]), tf(0.6, [1, 1, 1])*delay(τ)) # Feed 60% of the output back at the input with a delay of 2 seconds (like an echo)
Pd = c2d(Pc, Ts)
res = lsim(Pd, ref')
decho = iddata(res)
plot(bodeplot(Pd), pzmap(Pd), plot(decho))
```

The model-selection plot below indicates that we need to reach model orders of 24 to get a good fit
```@example DELAY
find_nanb(decho, 3:30, 5:30, xrotation=90, size=(800, 300), margin=5Plots.mm, xtickfont=6)
```

trying to estimate a 4:th order model with input delay of 20 samples does not work at all this time, but fitting a 24:th order model does
```@example DELAY
model3 = arx(decho, 5, 5, inputdelay=τ_samples)
model4 = subspaceid(decho, 24)
figsim = simplot(model3, decho, zeros(ss(model3).nx), sysname="ARX")
simplot!(model4, decho, zeros(model4.nx), ploty=false, plotu=false, sysname="Subspace")
```

Keep in mind that we do not add any disturbance in our simualtions here, and estimating 24:th order models is likely going to be a challenging task in practice.