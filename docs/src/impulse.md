# Impulse-response estimation
The functions `impulseest(h,y,u,order)` and [`impulseestplot`](@ref) performs impulse-response estimation by fitting a high-order FIR model.

Example
```julia
T = 200
h = 1
t = h:h:T
sim(sys,u) = lsim(sys, u, t)[1][:]
sys = c2d(tf(1,[1,2*0.1,0.1]),h)

u  = randn(length(t))
y  = sim(sys, u)
d  = iddata(y,u,h)

impulseestplot(d,50, lab="Estimate")
impulseplot!(sys,50, lab="True system")
```
![window](../../figs/impulse.svg)

See the [example notebooks](
https://github.com/JuliaControl/ControlExamples.jl) for more details.

```@docs
ControlSystemIdentification.impulseest
ControlSystemIdentification.impulseestplot
```