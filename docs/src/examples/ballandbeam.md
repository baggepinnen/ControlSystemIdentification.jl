In this example, we will estimate a model for a ball on a beam. 

We will get the data from [STADIUS's Identification Database](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html)

```@example ballbeam
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystems, RobustAndOptimalControl


## Ball and beam
url = "https://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/ballbeam.dat.gz"
zipfilename = "/tmp/bb.dat.gz"
path = Base.download(url, zipfilename)
run(`gunzip -f $path`)
data = readdlm(path[1:end-3])
u = data[:, 1]' # beam angle
y = data[:, 2]' # ball position
d = iddata(y, u, 0.1)
```
The input consists of the beam angle and the output is the position of the ball on the beam. This process is unstable (indeed, any student who has ever tried to control this process is familiar with the very recognizable sound of a nickel ball hitting the floor).

Before we estimate any model, we inspect the data and the coherence function
```@example ballbeam
plot(
    plot(d),
    coherenceplot(d),
)
```
The coherence is low for very low and high frequencies. Since the process is unstable, the data is collected in closed loop, and the input does not contain much DC energy. We thus expect to have difficulties recovering the DC properties of the model.

Since the data is collected in closed loop, we use an identification method that is unbiased in the presence of feedback. We'll go with the prediction-error method (PEM).
Since the process is unstable, we tell the identification routine that we accept an unstable model by saying `stable=false`. If we do not do this, [`newpem`](@ref) will try to stabilize an estimated unstable model. 

We also split the data in half, and use the first half for estimation and the second for validation.
```@example ballbeam
dtrain = d[1:end÷2]
dval = d[end÷2:end]

# A model of order 2-3 is reasonable, 
model,_ = newpem(dtrain, 3, stable=false)

predplot(model, dval, h=1)
predplot!(model, dval, h=10, ploty=false)
predplot!(model, dval, h=20, ploty=false)
```
The figures above show the result of predicting $h=\left{1, 10, 20\right}$ steps into the future. Since the process is unstable, simulation is not feasible, and already 20 steps prediction shows tendencies towards being unstable.

We can visualize the estimated models in the frequency domain as well. We show both the model estimated using PEM and a nonparametric estimate using a Fourier-based method ([`tfest`](@ref)), this method estimates a noise model as well.

```@example ballbeam
w = exp10.(LinRange(-1.5, log10(pi/d.Ts), 200))
bodeplot(model.sys, w, lab="PEM", plotphase=false)
plot!(tfest(d))
```
It looks like the two models disagree for low frequencies, which is expected after the discussion above.