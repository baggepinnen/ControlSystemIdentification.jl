In this example, we will estimate a model for a laboratory setup acting like a hair dryer. Air is fanned through a tube and heated at the inlet. The air temperature is measured by a thermocouple at the output. The input is the voltage over the heating device (a mesh of resistor wires).

The example comes from [STADIUS's Identification Database](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html) 

> Ljung L.  System identification - Theory for the User. Prentice Hall, Englewood Cliffs, NJ, 1987. 

```@example dryer
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystems

url = "https://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/dryer.dat.gz"
zipfilename = "/tmp/dryer.dat.gz"
path = Base.download(url, zipfilename)
run(`gunzip -f $path`)
data = readdlm(path[1:end-3])
u = data[:, 1]' # voltage
y = data[:, 2]' # air temp
d = iddata(y, u, 0.08) # sample time not specified for data, 0.08 is a guess
```
The input consists of the voltage to the heating element and the output is the air temperature

Before we estimate any model, we inspect the data and the coherence function
```@example dryer
plot(
    plot(d),
    coherenceplot(d),
)
```
The coherence looks good up until about 10 rad/s, above this frequency we can not trust the model. We notice that the data is not zero centered, and thus remove the mean before continuing (remember to correctly handle the operating point when controlling a process). If we look at an estimated impulse response
```@example dryer
d = detrend(d)
impulseestplot(d, 40, σ=3)
```
we see that there is a three step input delay. 

Before we estimate models, we split the data in a estimation set and a validation set. We then estimate two different models, one using hte prediction-error method ([`newpem`](@ref)) and one using standard least-squares ([`arx`](@ref)). When we estimate the ARX model, we tell the estimator that we have a known input delay of 3 samples. We then validate the estimated models by using them for prediction and simulation on the validation data.
```@example dryer
dtrain = d[1:end÷2]
dval = d[end÷2:end]
## A model of order 3 is reasonable.
model_pem,_ = newpem(dtrain, 3)
model_arx = arx(dtrain, 2, 2, inputdelay=3)

predplot(model_pem, dval, sysname="PEM")
predplot!(model_arx, dval, ploty=false, sysname="ARX")
simplot!(model_pem, dval, ploty=false, sysname="PEM")
simplot!(model_arx, dval, ploty=false, sysname="ARX")
```
The models are roughly comparable, where the PEM model is slightly better at prediction while the ARX model is slightly better at simulation. Note, [`newpem`](@ref) takes a keyword argument `focus` that can be set to `focus = :simulation` in order to improve simulation performance.

To finalize, we compare the models in the frequency domain to a nonparametric estimate of the transfer function made with Fourier-based methods ([`tfest`](@ref)):
```@example dryer
w = exp10.(LinRange(-1, log10(pi/d.Ts), 200))
bodeplot(model_pem.sys, w, lab="PEM", plotphase=false)
bodeplot!(model_arx, w, lab="ARX", plotphase=false)
plot!(tfest(d))
```
The two parametric models are quite similar and agree well with the nonparametric estimate. We also see that the nonparametric estimate becomes rather noisy above 10 rad/s, something we could predict based on the coherence function.

We can compare the impulse responses of the estimated model to the impulse response that was estimated directly from data above:
```@example dryer
impulseestplot(d, 40, σ=3, lab="Data", seriestype=:steppost)
plot!(impulse(model_pem, 3), lab="PEM")
plot!(impulse(model_arx, 3), lab="ARX")
```
The ARX model has an impulse response that is exactly zero for the first three samples since we indicated `inputdelay=3` when estimating this model. The PEM model did not know this, but figured it out from the data nevertheless.

As a last step of validation, we perform residual analysis. If a model has extracted all available useful information from the data, the residuals should form a white-noise sequence, and there should be no correlation between the input and the residuals. To this end, we have the function [`residualplot`](@ref):
```@example dryer
residualplot(model_pem, dval, lab="PEM")
residualplot!(model_arx, dval, lab="ARX")
```
As we can see, there is some slight correlation left in the residuals, the dashed black lines show 95% significance levels. This small amount of correlation is usually nothing to worry about if the model fit is high, i.e., the residuals are small, and we'll thus consider ourselves done at this point.
