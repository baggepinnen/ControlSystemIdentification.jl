# Delay estimation
A frequent property of control systems is the presence of _delays_, either due to processing time, network latency, or other physical phenomena. Delays that occur internally in the system will in the discrete-time setting add a potentially large number of states to the system, ``\tau / T_s`` state variables are required to represent a delay of ``\tau`` seconds in a discrete-time system with sampling time ``T_s``. A common special case is that the delay occurs at either the input or the output of the system, e.g, due to communication delays. Estimating this delay ahead of the estimation of the model is often beneficial, since it reduces the number of parameters that need to be estimated. Below, we generate a dataset with a large input delay ``\tau`` and have a look at how we can estimate ``\tau``.

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
Estimation of impulse responses in the presence of such large delays is numerically challenging, and a regularization of ``λ=0.5`` was required to achieve a reasonable result. If the delay is expected to be large, it is thus recommended to use the cross-correlation method instead, since we cannot tune the regularization parameter ``λ`` in a practical setting when we don't know what impulse response to expect. However, for short delays the impulse-response method works rather well.


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
τ_samples = 20
d2 = iddata(d.y[:, τ_samples+1:end], d.u[:, 1:end-τ_samples], d.Ts)
crosscorplot(d2)
```
The cross-correlation plot now indicates that the delay is gone. Please note, one sample delay is most of the time expected, indeed, it's unrealistic for the output of a physical system to respond at the same time as something happens at the input. Once we have estimated a model with the dataset with the delay removed, we must add the delay back into the estimated model, e.g.,
```@example DELAY
Pest = subspaceid(d2, focus=:simulation) # Estimate a model without delay
tf(d2c(Pest)) # Does it match the original continuous-time system without delay?
```
The estimated model above should be very close to the system `tf(1, [1, 0.5, 1])` (some small higher-order terms in the numerator are expected). To add the delay to this model, we do
```@example DELAY
Pest_τ = Pest*delay(τ, Ts)
bodeplot([P, Pest_τ], lab=["True system" "Estimated system"])
```
The estimated model should now match the true system very well, including the large drop in phase for higher frequencies due to the delay.