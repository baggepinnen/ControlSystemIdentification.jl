In this example, we will estimate a model for a four-stage evaporator to reduce the water content of a product, for example milk. The 3 inputs are feed flow, vapor flow to the first evaporator stage and cooling water flow. The three outputs are the dry matter content, the flow and the temperature of the outcoming product.

The example comes from [STADIUS's Identification Database](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html) 

> Zhu Y., Van Overschee P., De Moor B., Ljung L.,
> Comparison of three classes of identification methods. Proc. of SYSID '94, 

```@example furnace
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystems, RobustAndOptimalControl

url = "https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/evaporator.dat.gz"
zipfilename = "/tmp/evaporator.dat.gz"
path = Base.download(url, zipfilename)
run(`gunzip -f $path`)
data = readdlm(path[1:end-3])
# Inputs:
# 	u1: feed flow to the first evaporator stage
# 	u2: vapor flow to the first evaporator stage
# 	u3: cooling water flow
# Outputs:
# 	y1: dry matter content
# 	y2: flow of the outcoming product
# 	y3: temperature of the outcoming product
u = data[:, 1:3]'  
y = data[:, 4:6]'
d = iddata(y, u, 1) 
```
The input consists of two heating inputs and one cooling input, while there are 6 outputs from temperature sensors in a cross section of the furnace.

Before we estimate any model, we inspect the data
```@example furnace
plot(d, layout=6)
```

We split the data in two, and use the first part for estimation and the second for validation. A model of order around 8 is reasonable (the paper uses 6-13). This system requires zeroD=false to be able to capture a direct feedthrough, otherwise the fit will always be rather poor.
```@example furnace
dtrain = d[1:end÷2]
dval = d[end÷2:end]

model,_ = newpem(dtrain, 8, zeroD=false)
```

```@example furnace
predplot(model, dval, h=1, layout=d.ny)
predplot!(model, dval, h=5, ploty=false)
```
The figures above show the result of predicting $h=\left{1, 5\right}$ steps into the future.

We can visualize the estimated model in the frequency domain as well. 
```@example furnace
w = exp10.(LinRange(-2, log10(pi/d.Ts), 200))
sigmaplot(model.sys, w, lab="PEM", plotphase=false)
```

Let's compare prediction performance to the paper
```@example furnace
ys = predict(model, dval, h=5)
ControlSystemIdentification.mse(dval.y-ys)
```
The authors got the following errors: [0.24, 0.39, 0.14]

