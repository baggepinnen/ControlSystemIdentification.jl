A typical model for a temperature-controlled system is 
```math
\tau \dot T = -T  + Bu + c
```
where $T$ is the temperature, $u$ the control signal and $c$ a constant offset, e.g., related to the temperature surrounding the controlled system. The time constant $\tau$ captures the relation between stored energy and the resistance to heat flow and determines how fast the temperature is changing. This system can be written on transfer-function form like (omitting $c$)
```math
\dfrac{B}{\tau s + 1}U(s)
```
This is a simple first-order transfer function which can be estimated with, e.g., the functions [`arx`](@ref) or [`plr`](@ref). To illustrate this, we create such a system and simulate some data from it.
```@example temp
using ControlSystemsBase, ControlSystemIdentification, Plots
w = 2pi .* exp10.(LinRange(-3, log10(0.5), 500))
G0 = tf(1, [10, 1]) # The true system, 10ẋ = -x + u
G = c2d(G0, 1)      # discretize with a sample time of 1s
println("True system")
display(G0)

u = sign.(sin.((0:0.01:20) .^ 2))' # sample a control input for identification
y, t, x = lsim(ss(G), u) # Simulate the true system to get test data
yn = y .+ 0.2 .* randn.() # add measurement noise
data = iddata(yn, u, t[2] - t[1]) # create a data object
plot(data)
```

We see that the data we're going to use for identification is a chirp input. Chirps are excellent for identification as they have a well defined and easily controllable interval of frequencies for identification. We start by inspecting the coherence plot to ensure that the data is suitable for identification of a linear system
```@example temp
coherenceplot(data, hz=true)
```
The coherence is high for all frequencies spanned by the chirp, after which it drops significantly. This implies that we can only ever trust the identified model to be accurate up to the highest frequency that was present in the chirp input.

Next we set the parameters for the estimation, the numerator and denominator have one parameter each, so we set $n_a = n_b = 1$ and estimate two models.
```@example temp
na, nb = 1, 1 # number of parameters in denominator and numerator
Gh = arx(data, na, nb, estimator = wtls_estimator(data.y, na, nb)) # estimate an arx model
Gh2, noise_model = plr(data, na, nb, 1) # try another identification method

Gh, Gh2
```
Least-squares estimation of ARX models from data with high measurement noise is known to lead to models with poor low-frequency fit, we therefore used the `wtls_estimator(data.y, na, nb)` which performs the estimation with total-least squares.

We can plot the results in several different ways:
```@repl temp
# Plot results
println("Estimated system in continuous time")
display(d2c(Gh)) # Convert from discrete to continuous time
```

```@example temp
bp = bodeplot(G, w, lab = "G (true)", hz = true, l = 5)
bodeplot!(Gh, w, lab = "arx", hz = true)
bodeplot!(Gh2, w, lab = "plr", hz = true, ticks = :default)

sp = plot(step(G, 150), lab="G (true)")
plot!(step(Gh, 150), lab = "arx")
plot!(step(Gh2, 150), lab = "plr", ticks = :default)
hline!([1], primary = false, l = (:black, :dash))

lp = plot(lsim(ss(G), u), lab="G (true)")
plot!(lsim(ss(Gh), u), lab = "arx")
plot!(lsim(ss(Gh2), u), lab = "plr", ticks = :default)
plot!(data.t, yn[:], lab = "Estimation data", alpha=0.3)

plot(bp, sp, lp, layout = @layout([[a b]; c]))
```