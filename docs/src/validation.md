

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
nothing # hide
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



## Different length predictors
When the prediction horizon gets longer, the mapping from $u -> ŷ$ approaches that of the simulation system, while the mapping $y -> ŷ$ gets smaller and smaller.
```@example validation
using LinearAlgebra
G   = c2d(DemoSystems.resonant(), 0.1)
K   = kalman(G, I(G.nx), I(G.ny))
sys = add_input(G, K, I(G.ny)) # Form an innovation model with inputs u and e

T = 10000
u = randn(G.nu, T)
e = 0.1randn(G.ny, T)
y = lsim(sys, [u; e]).y
d = iddata(y, u, G.Ts)
Gh,_ = newpem(d, G.nx, zeroD=true)

# Create predictors with different horizons
p1   = observer_predictor(Gh)
p2   = observer_predictor(Gh, h=2)
p10  = observer_predictor(Gh, h=10)
p100 = observer_predictor(Gh, h=100)

bodeplot([p1, p2, p10, p100], plotphase=false, lab=["1" "" "2" "" "10" "" "100" ""])
bodeplot!(sys, ticks=:default, plotphase=false, l=(:black, :dash), lab=["sim" ""], title=["From u" "From y"])
```

The prediction error as a function of prediction horizon approaches the simulation error.
```@example validation
using Statistics
hs = [1:40; 45:5:80]
perrs = map(hs) do h
    yh = predict(Gh, d; h)
    ControlSystemIdentification.rms(d.y - yh) |> mean
end
serr = ControlSystemIdentification.rms(d.y - simulate(Gh, d)) |> mean

plot(hs, perrs, lab="Prediction errors", xlabel="Horizon", ylabel="RMS error")
hline!([serr], lab="Simulation error", l=:dash, legend=:bottomright, ylims=(0, Inf))
```

