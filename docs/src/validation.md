

# Validation
A number of functions are made available to assist in validation of the estimated models. We illustrate by an example

Generate some test data:
```@example validation
using ControlSystemIdentification, ControlSystems, Random
using ControlSystemIdentification: newpem
Random.seed!(1)
T          = 200
nx         = 2
nu         = 1
ny         = 1
x0         = randn(nx)
σy         = 0.5
sim(sys,u) = lsim(sys, u, 1:T)[1]
sys        = tf(1, [1, 2*0.1, 0.1])
sysn       = tf(σy, [1, 2*0.1, 0.3])
# Training data
u          = randn(nu,T)
y          = sim(sys, u)
yn         = y + sim(sysn, randn(size(u)))
dn         = iddata(yn, u, 1)
# Validation data
uv         = randn(nu, T)
yv         = sim(sys, uv)
ynv        = yv + sim(sysn, randn(size(uv)))
dv         = iddata(yv, uv, 1)
dnv        = iddata(ynv, uv, 1)
```
We then fit a couple of models
```@example validation
res = [newpem(dn, nx, focus=:prediction) for nx = [1,3,4]];
```
After fitting the models, we validate the results using the validation data and the functions `simplot` and `predplot` (cf. Matlab sys.id's `compare`):
```@example validation
using Plots
ω   = exp10.(range(-2, stop=log10(pi), length=150))
fig = plot(layout=4, size=(1000,600))
for i in eachindex(res)
    sysh, x0h, opt = res[i]
    simplot!( sysh,dnv,x0h; subplot=1, ploty=i==1)
    predplot!(sysh,dnv,x0h; subplot=2, ploty=i==1)
end
bodeplot!((getindex.(res,1)),                     ω, plotphase=false, subplot=3, title="Process", linewidth=2*[4 3 2 1])
bodeplot!(innovation_form.(getindex.(res,1)),     ω, plotphase=false, subplot=4, linewidth=2*[4 3 2 1])
bodeplot!(sys,                                    ω, plotphase=false, subplot=3, lab="True", l=(:blue, :dash), legend = :bottomleft, title="System model")
bodeplot!(innovation_form(ss(sys),syse=ss(sysn)), ω, plotphase=false, subplot=4, lab="True", l=(:blue, :dash), ylims=(0.1, 100), legend = :bottomleft, title="Noise model")
```

In the figure, simulation output is compared to the true model on the top left and prediction on top right. The system models and noise models are visualized in the bottom plots. Both high-order models capture the system dynamics well, but struggle slightly with capturing the gain of the noise dynamics.
The figure also indicates that a model with 4 poles performs best on both prediction and simulation data. The true system has 4 poles (two in the process and two in the noise process) so this is expected. However, the third order model performs almost equally well and may be a better choice.


Prediction models may also be evaluated using a `h`-step prediction, here `h` is short for "horizon".
```@example validation
figh = plot()
for i in eachindex(res)
    sysh, x0h, opt = res[i]
    predplot!(sysh, dnv, x0h, ploty=i==1, h=5)
end
figh
```


See also [`simulate`](@ref), [`predplot`](@ref), [`simplot`](@ref), [`coherenceplot`](@ref)
