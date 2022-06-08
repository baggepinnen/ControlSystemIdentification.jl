In this example, we will estimate a model for a flexible robot arm. 

We will get the data from [STADIUS's Identification Database](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html)

```@example robot
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystems

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
xt = [3,10,30,100]
plot(
    plot(d),
    coherenceplot(d, xticks=(xt,xt)),
)
```
The coherence is low for high frequencies as well as frequencies between 11 and 40 rad/s. We should thus be careful with relying on the estimated model too much in these frequency ranges. The reason for the low coherence may be either a poor signal-to-noise ratio, or the presence of nonlinearities. For systems with anti-resonances, like this one, the SNR is often poor at the notch frequencies (indeed, a notch frequency is defined as a frequency where there will be very little signal).

We also split the data in half, and use the first half for estimation and the second for validation. We'll use a subspace-based identification algorithm for estimation
```@example robot
dtrain = d[1:end÷2]
dval = d[end÷2:end]

# A model of order 4 is reasonable. Double-mass model.
model = subspaceid(dtrain, 4)

predplot(model, dval, h=1)
predplot!(model, dval, h=10, ploty=false)
simplot!(model, dval, ploty=false)
```
The figures above show the result of predicting $h={1, 10, \infty}$ steps into the future.

We can visualize the estimated models in the frequency domain as well. We show both the model estimated using PEM and a nonparametric estimate using a Fourier-based method ([`tfest`](@ref)), this method estimates a noise model as well.

```@example robot
w = exp10.(LinRange(-1, log10(pi/d.Ts), 200))
bodeplot(model.sys, w, lab="PEM", plotphase=false)
plot!(tfest(d), legend=:bottomleft)
```
It looks like the model fails to capture the notches accurately. Estimating zeros is known to be hard, both in practice and in theory.
