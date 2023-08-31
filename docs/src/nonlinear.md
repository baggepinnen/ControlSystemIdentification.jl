# Identification of nonlinear models

For nonlinear graybox identification, i.e., identification of models on a known form but with unknown parameters, we suggest to implement a nonlinear version of the prediction-error method by using an Unscented Kalman filter as predictor. A tutorial for this is available [in the documentation of LowLevelParticleFilters.jl](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Using-an-optimizer) which also provides the UKF.


This package provides very elementary identification of nonlinear systems on Hammerstein-Wiener form, i.e., systems with a static input nonlinearity and a static output nonlinearity with a linear system inbetween, **where the nonlinearities are known**. The only aspect of the nonlinearities that are optionally estimated are parameters.

The procedure to estimate such a model is detailed in the docstring for [`newpem`](@ref).

The result of this estimation is the linear system _without_ the nonlinearities.

The default optimizer BFGS may struggle with problems including nonlinearities, if you do not get good results, try a different optimizer, e.g., `optimizer = Optim.NelderMead()`.


## Example 1:

The example below identifies a model of a resonant system with a static where the sign of the output is unknown, i.e., the output nonlinearity is given by ``y_{nl} = |y|``. To make the example a bit more realistic, we also simulate colored measurement and input noise, `yn` and `un`.
```@example HW
using ControlSystemIdentification, ControlSystemsBase
using ControlSystemsBase.DemoSystems: resonant
using Random, Plots

# Generate some data from the system
Random.seed!(1)
T = 200
sys = c2d(resonant(ω0 = 0.1) * tf(1, [0.1, 1]), 1)# generate_system(nx, nu, ny)
nx = sys.nx
nu = 1
ny = 1
x0 = zeros(nx)
sim(sys, u, x0 = x0) = lsim(sys, u, 1:T, x0 = x0)[1]
sysn = c2d(resonant(ω0 = 1) * tf(1, [0.1, 1]), 1)

σu = 1e-2 # Input noise standard deviation
σy = 1e-3 # Output noise standard deviation

u  = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)

# Nonlinear output transformation
ynn = abs.(yn)
d  = iddata(ynn, un, 1)
output_nonlinearity = (y, p) -> y .= abs.(y)

# Estimate 10 models with different random initialization and pick the best one
# If results are poor, try `optimizer = Optim.NelderMead()` instead
results = map(1:10) do _
    sysh, x0h, opt = newpem(d, nx; output_nonlinearity, show_trace=false, focus = :simulation)
    (; sysh, x0h, opt)
end;

(; sysh, x0h, opt) = argmin(r->r.opt.minimum, results) # Find the model with the smallest cost

yh = simulate(sysh, d, x0h)
output_nonlinearity(yh, nothing) # We need to manually apply the output nonlinearity to the prediction
plot(d.t, [abs.(y); u]', lab=["True nonlinear output" "Input"], seriestype = [:line :steps], layout=(2,1), xlabel="Time")
scatter!(d.t, ynn', lab="Measured nonlinear output", sp=1)
plot!(d.t, yh', lab="Simulation", sp=1, l=:dash)
```

## Example 2: 
Below, we identify a similar model but this time with data recorded from a physical system. The data comes from the belt-drive system depicted below.

![Belt drive](https://lh6.googleusercontent.com/AjBmg1ezDWGGMEX6f4vDJCpHFIM2PrAMZRzYLj6dA5033LYuhwU4O0NtwD_ZEhIYRtn2k0YX86nGMCfqrznY2apE5KmlrTZhhCV7rd6EbiNTjJbT=w1280)

The system is described in detail in [this report](http://www.google.com/url?q=http%3A%2F%2Fwww.it.uu.se%2Fresearch%2Fpublications%2Freports%2F2017-024%2F2017-024-nc.pdf&sa=D&sntz=1&usg=AOvVaw0yNPLBveaHDGWB9mwnHCxd) and the data is available on the link downloaded in the code snippet below.

The speed sensor available in this system cannot measure the direction, we thus have an absolute-value nonlinearity at the output similar to above. The technical report further indicates that there is a low-pass filter on the output, _after_ the nonlinearity. We do not have capabilities of estimating this complicated structure in this package, so we ignore the additional low-pass filter and only estimate only the initial linear system and the nonlinearity.

```@example beltdrive
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystemsBase

url = "http://www.it.uu.se/research/publications/reports/2017-024/CoupledElectricDrivesDataSetAndReferenceModels.zip"
zipfilename = "/tmp/bd.zip"
cd("/tmp")
path = Base.download(url, zipfilename)
@show run(`unzip -o $path`)
@show pwd()
@show run(`ls`)
data = readdlm("/tmp/DATAUNIF.csv", ',')[2:end, 1:4]
iddatas = map(0:1) do ind
    u = data[:, 1 + ind]' .|> Float64 # input
    y = data[:, 3 + ind]' .|> Float64 # output
    iddata(y, u, 1/50)
end

plot(plot.(iddatas)...)

d = iddatas[1] # We use one dataset for estimation 
output_nonlinearity = (y, p) -> y .= abs.(y)

nx = 3 # Model order

results = map(1:40) do _ # This example is a bit more difficult, so we try more random initializations
    sysh, x0h, opt = newpem(d, nx; output_nonlinearity, show_trace=true, focus=:simulation)
    (; sysh, x0h, opt)
end;

(; sysh, x0h, opt) = argmin(r->r.opt.minimum, results) # Find the model with the smallest cost

dv = iddatas[2] # We use the second dataset for validation
yh = simulate(sysh, dv)
output_nonlinearity(yh, nothing) # We need to manually apply the output nonlinearity to the simulation
plot(dv, lab=["Measured nonlinear output" "Input"], layout=(2,1), xlabel="Time")
plot!(dv.t, yh', lab="Simulation", sp=1, l=:dash)
```

```@example beltdrive
bodeplot(sysh)
```

If everything went as expected, the model should be able to predict the output reasonably well, and the estimated model should have a resonance peak around 20rad/s (compare with Fig. 8 in the report).

The dataset consists of two different experiments, here, we used one for identification and another one for validation. They both differ in the amplitude of the input. Ideally, we'd use a dataset that mixes different amplitudes for training.