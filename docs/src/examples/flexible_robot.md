In this example, we will estimate a model for a flexible robot arm. 

We will get the data from [STADIUS's Identification Database](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html)

```@example robot
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystemsBase
gr(fmt=:png) # hide

url = "https://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/robot_arm.dat.gz"
zipfilename = "/tmp/flex.dat.gz"
path = Base.download(url, zipfilename)
run(`gunzip -f $path`)
data = readdlm(path[1:end-3])
u = data[:, 1]' # torque
y = data[:, 2]' # acceleration
d = iddata(y, u, 0.01) # sample time not specified for data, 0.01 is a guess
```
The input consists of the motor torque and the output is the acceleration of the arm. 

Before we estimate any model, we inspect the data and the coherence function
```@example robot
xt = [2,5,10,20,50]
plot(
    plot(d),
    coherenceplot(d, xticks=(xt,xt), hz=true),
)
```
The coherence is low for high frequencies as well as frequencies between 2 and 6 Hz. We should thus be careful with relying on the estimated model too much in these frequency ranges. The reason for the low coherence may be either a poor signal-to-noise ratio, or the presence of nonlinearities. For systems with anti-resonances, like this one, the SNR is often poor at the notch frequencies (indeed, a notch frequency is defined as a frequency where there will be very little signal).
We can investigate the spectra of the input and output using [`welchplot`](@ref) (see also [`specplot`](@ref))
```@example robot
welchplot(d, yticks=(xt,xt))
```
Not surprisingly, we see that the input has very little power above 20Hz, this is the reason for the low coherence above 20Hz. Limiting the bandwidth of the excitation signal is usually a good thing, mechanical structures often exhibit higher-order resonances and nonlinear behavior at high frequencies, which we see eveidence of in the output spectrum at around 34Hz.

We also split the data in half, and use the first half for estimation and the second for validation. We'll use a subspace-based identification algorithm for estimation
```@example robot
dtrain = d[1:end÷2]
dval = d[end÷2:end]

# A model of order 4 is reasonable, a double-mass model. We estimate two models, one using subspace-based identification and one using the prediction-error method
using Optim
model_ss = subspaceid(dtrain, 4, focus=:prediction)
model_pem, x0_pem = newpem(dtrain, 4; focus=:prediction, optimizer=NelderMead(), iterations=500000, show_every=50000)

predplot(model_ss, dval, h=1)
predplot!(model_ss, dval, h=5, ploty=false)
simplot!(model_ss, dval, ploty=false)
```
The figures above show the result of predicting $h={1, 10, \infty}$ steps into the future.

We can visualize the estimated models in the frequency domain as well. We show both the model estimated using PEM and a nonparametric estimate using a Fourier-based method ([`tfest`](@ref)), this method estimates a noise model as well.

```@example robot
w = exp10.(LinRange(-1, log10(pi/d.Ts), 200))
bodeplot(model_pem.sys, w, lab="PEM", plotphase=false, hz=true)
bodeplot!(model_ss.sys, w, lab="Subspace", plotphase=false, hz=true)
plot!(tfest(d), legend=:bottomleft, hz=true, xticks=(xt,xt))
```
It looks like the model fails to capture the notches accurately. Estimating zeros is known to be hard, both in practice and in theory. In the estimated disturbance (labeled *Noise*), we see a peak at around 34Hz. This is likely an overtone due to nonlinearities.

We can also investigate how well the models predict for various prediction horizons, and compare that to how well the model does in open loop (simulation)
```@example robot
using Statistics
hs = [1:40; 45:5:80]
perrs_pem = map(hs) do h
    yh = predict(model_pem, d, x0_pem; h)
    ControlSystemIdentification.rms(d.y - yh) |> mean
end
perrs_ss = map(hs) do h
    yh = predict(model_ss, d; h)
    ControlSystemIdentification.rms(d.y - yh) |> mean
end
serr_pem = ControlSystemIdentification.rms(d.y - simulate(model_pem, d)) |> mean
serr_ss = ControlSystemIdentification.rms(d.y - simulate(model_ss, d)) |> mean

plot(hs, perrs_pem, lab="Prediction errors PEM", xlabel="Prediction Horizon", ylabel="RMS error")
plot!(hs, perrs_ss, lab="Prediction errors Subspace")
hline!([serr_pem], lab="Simulation error PEM", l=:dash, c=1, ylims=(0, Inf))
hline!([serr_ss], lab="Simulation error Subspace", l=:dash, c=2, legend=:bottomright, ylims=(0, Inf))
```
We see that the prediction-error model does well at prediction few-step predictions (indeed, this is what PEM optimizes), while the model identified using `subspaceid` does better in open loop.
The simulation performance can be improved upon further by asking for `focus=:prediction` when the models are estimated.
