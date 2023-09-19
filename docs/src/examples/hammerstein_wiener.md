## Hammerstein-Wiener estimation of nonlinear belt-drive system
In this example, we identify a Wiener model (output nonlinearity only) with data recorded from the belt-drive system depicted below.

![Belt drive](https://user-images.githubusercontent.com/3797491/264962931-e62c56ee-3dab-43f5-bdd3-858c841fb516.png)

The system is described in detail in [this report](http://www.google.com/url?q=http%3A%2F%2Fwww.it.uu.se%2Fresearch%2Fpublications%2Freports%2F2017-024%2F2017-024-nc.pdf&sa=D&sntz=1&usg=AOvVaw0yNPLBveaHDGWB9mwnHCxd) and the data is available on the link downloaded in the code snippet below.

The speed sensor available in this system cannot measure the direction, we thus have an absolute-value nonlinearity at the output. The technical report further indicates that there is a low-pass filter on the output, _after_ the nonlinearity. We do not have capabilities of estimating this complicated structure in this package, so we ignore the additional low-pass filter and estimate only the initial linear system and the nonlinearity.

The estimation of the Wiener model is performed using the [`newpem`](@ref) function, see [Identification of nonlinear models](@ref) for more details.

```@example beltdrive
using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystemsBase

url = "http://www.it.uu.se/research/publications/reports/2017-024/CoupledElectricDrivesDataSetAndReferenceModels.zip"
zipfilename = "/tmp/bd.zip"
cd("/tmp")
path = Base.download(url, zipfilename)
run(`unzip -o $path`)
data = readdlm("/tmp/DATAUNIF.csv", ',')[2:end, 1:4]
iddatas = map(0:1) do ind
    u = data[:, 1 + ind]' .|> Float64 # input
    y = data[:, 3 + ind]' .|> Float64 # output
    iddata(y, u, 1/50)
end

plot(plot.(iddatas)...)

d = iddatas[1] # We use one dataset for estimation 
coherenceplot(d)
```
The [`coherenceplot`](@ref), a measure of how well a linear model describes the relation between input and output, unsurprisingly indicates that the system is nonlinear. Before estimating a linear model, it's good practice to inspect this non-parametric measure of linearity.


```@example beltdrive
output_nonlinearity = (y, p) -> y .= abs.(y)

nx = 3 # Model order

results = map(1:40) do _ # This example is a bit more difficult, so we try more random initializations
    sysh, x0h, opt = newpem(d, nx; output_nonlinearity, show_trace=false, focus=:simulation)
    (; sysh, x0h, opt)
end;

(; sysh, x0h, opt) = argmin(r->r.opt.minimum, results) # Find the model with the smallest cost

dv = iddatas[2] # We use the second dataset for validation
yh = simulate(sysh, dv, x0h)
output_nonlinearity(yh, nothing) # We need to manually apply the output nonlinearity to the simulation
plot(dv, lab=["Measured nonlinear output" "Input"], layout=(2,1), xlabel="Time")
plot!(dv.t, yh', lab="Simulation", sp=1, l=:dash)
```

```@example beltdrive
bodeplot(sysh)
```

If everything went as expected, the model should be able to predict the output reasonably well, and the estimated model should have a resonance peak around 20rad/s (compare with Fig. 8 in the report).

The dataset consists of two different experiments. In this case we used one for identification and another one for validation. The experiments differ in the amplitude of the input. Ideally, we would use a dataset that combines different amplitudes for training.