# Impulse-response estimation
The functions [`impulseest`](@ref)`(data, order; λ=0, estimator=ls)` and [`impulseestplot`](@ref) perform impulse-response estimation by fitting a high-order FIR model. The function [`okid`](@ref) estimates Markov parameters and is applicable to MIMO systems.

SISO example
```julia
T = 200
h = 1
t = 0:h:T-h
sys = c2d(tf(1,[1,2*0.1,0.1]),h)

u  = randn(1, length(t))
res = lsim(sys, u, t)
d  = iddata(res)

impulseestplot(d,50, lab="Estimate", seriestype=:steppost)
plot!(impulse(sys,50), lab="True system", l=:dash)
```
![window](https://github.com/baggepinnen/ControlSystemIdentification.jl/blob/master/figs/impulse.svg?raw=true)


MIMO example
```@example
using ControlSystemIdentification, ControlSystems, Plots
T = 200
h = 1
t = 0:h:T-h
sys = ssrand(2,2,4, proper=true, Ts=h)

u  = randn(sys.nu, length(t))
res = lsim(sys, u, t)
d  = iddata(res)

H = okid(d, sys.nx)
plot(impulse(sys,50), lab="True system", layout=sys.ny+sys.nu, sp=(1:4)')
plot!(reshape(H, sys.nu+sys.ny, :)', lab="OKID Estiamte", seriestype=:steppre, l=:dash)
```

See the [example notebooks](
https://github.com/JuliaControl/ControlExamples.jl) for more details.
