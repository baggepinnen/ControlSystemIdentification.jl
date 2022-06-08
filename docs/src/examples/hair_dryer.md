In this example, we will estimate a model for a laboratory setup acting like a hair dryer. Air is fanned through a tube and heated at the inlet. The air temperature is measured by a thermocouple at the output. The input is the voltage over the heating device (a mesh of resistor wires).

The example comes from [STADIUS's Identification Database](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html) 

> Ljung L.  System identification - Theory for the User. Prentice Hall, Englewood Cliffs, NJ, 1987. 

```@example dryer
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystems, RobustAndOptimalControl

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
The models are roughly comparable, where the PEM model is slightly better at prediction while the ARX model is slightly better at simulation. Note, [`newpem`](@ref) takes a keyword argument `focus` that can be et to `focus = :simulation` in order to improve simulation performance.

To finalize, we compare the models in the frequency domain to a nonparametric estimate of the transfer function made with Fourier-based methods ([`tfest`](@ref)):
```@example dryer
w = exp10.(LinRange(-1, log10(pi/d.Ts), 200))
bodeplot(model_pem.sys, w, lab="PEM", plotphase=false)
bodeplot!(model_arx, w, lab="ARX", plotphase=false)
plot!(tfest(d))
```
The two parametric models are quite similar and agree well with the nonparametric estimate. We also see that the nonparametric estimate becomes rather noisy above 10 rad/s, something we could predict based on the coherence function.

