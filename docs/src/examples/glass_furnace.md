In this example, we will estimate a model for a glass furnace. 

We will get the data from [STADIUS's Identification Database](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html)

```@example furnace
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystemsBase
gr(fmt=:png) # hide

url = "https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/glassfurnace.dat.gz"
zipfilename = "/tmp/furnace.dat.gz"
path = Base.download(url, zipfilename)
run(`gunzip -f $path`)
data = readdlm(path[1:end-3])
u = data[:, 2:4]'  
y = data[:, 5:10]'
d = iddata(y, u, 1)
```
The input consists of two heating inputs and one cooling input, while there are 6 outputs from temperature sensors in a cross section of the furnace.

Before we estimate any model, we inspect the data
```@example furnace
plot(d, layout=9)
```

We split the data in two, and use the first part for estimation and the second for validation. This system requires `zeroD=false` to be able to capture a direct feedthrough to output 4, otherwise the fit for output 4 will always be rather poor.
```@example furnace
dtrain = d[1:2end÷3]
dval = d[2end÷3:end]

model = subspaceid(dtrain, 7, zeroD=false)
nothing # hide
```
We can have a look at the $D$ matrix in the estimated model
```@example furnace
model.D
```
indeed, the (4,3) element is rather large. 

We validate the model by prediction on the validation data:
```@example furnace
predplot(model, dval, h=1, layout=6)
predplot!(model, dval, h=10, ploty=false)
```
The figures above show the result of predicting $h={1, 10}$ steps into the future.

We can visualize the estimated model in the frequency domain as well. 
```@example furnace
w = exp10.(LinRange(-3, log10(pi/d.Ts), 200))
sigmaplot(model.sys, w, lab="MOESP")
```

