```@raw html
<p style="text-align:center">

<img src="https://avatars.githubusercontent.com/u/10605979?s=400&u=7b2efdd404c4db3b3f067f04c305d40c025a8961&v=4" alt="JuliaControl logo">

<br> 

<a class="github-button" href="https://github.com/baggepinnen/ControlSystemIdentification.jl" data-color-scheme="no-preference: light; light: light; dark: dark;" data-icon="octicon-star" data-show-count="true" aria-label="Star baggepinnen/ControlSystemIdentification.jl on GitHub">Star</a>

<script async defer src="https://buttons.github.io/buttons.js"></script>
</p> 
```

# ControlSystemIdentification

[![CI](https://github.com/baggepinnen/ControlSystemIdentification.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/ControlSystemIdentification.jl/actions)
[![codecov](https://codecov.io/gh/baggepinnen/ControlSystemIdentification.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/ControlSystemIdentification.jl)

System identification for [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl/). 

System identification is the process of estimating a dynamical model from data. This packages estimates primarily linear time-invariant (LTI) models, in the form of statespace systems
```math
\begin{aligned}
x^+ &= Ax + Bu + Ke\\
y &= Cx + Du + e
\end{aligned}
```
or in the form of transfer functions
```math
Y(z) = \dfrac{B(z)}{A(z)}U(z)
```

We also have capabilities for estimation of nonlinear Hammerstein-Wiener models and linear/nonlinear gray-box identification of models in continuous or discrete time.

This package is implemented in the free and open-source programming language [Julia](https://julialang.org/).

If you are new to this package, start your journey through the documentation by learning about [Identification data](@ref). Examples are provided in the Examples section and in the form of jupyter notebooks [here](https://github.com/JuliaControl/ControlExamples.jl). An introductory video is available below (system identification example starts around 55 minutes)

```@raw html
<iframe width="560" height="315" src="https://www.youtube.com/embed/Fdz2Fsm1aTY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

See also the [YouTube playlist](https://youtube.com/playlist?list=PLC0QOsNQS8ha6SwaNOZDXyG9Bj8MPbF-q&si=AiOZVhBVwYReDrAm) with tutorials using this package.

## Installation
Install [Julia](https://julialang.org/) from the [download](https://julialang.org/downloads/) page. Then, in the Julia REPL, type
```julia
using Pkg
Pkg.add("ControlSystemIdentification")
```

*Optional:* To work with linear systems and plot Bode plots etc., also install the control toolbox [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl) package which this package builds upon, as well as the plotting package
```julia
Pkg.add(["ControlSystemIdentification", "ControlSystemsBase", "Plots"])
```

## Algorithm overview
The following table indicates which estimation algorithms are applicable in different scenarios. A green circle (🟢) indicates that a particular method is well suited for the situation, an orange diamond (🔶) indicates that a match is possible, but somehow not ideal, while a red square (🟥) indicates that a method in its standard form is ill suited for the situation. The table is not exhaustive, and is intended to give a rough overview of the applicability of different algorithms.

```@setup ALGOVERVIEW
using PrettyTables, Markdown

header = ["Estimation method", "SIMO", "MISO", "Disturbance models", "Nonlinearities", "Custom loss", "Time domain", "Frequency domain", "Multiple dataset"]

data = [
    md"`newpem`"            "🟢" "🟢" "🟢" "🟢" "🟢" "🟢" "🟥" "🟥"
    md"`subspaceid`"        "🟢" "🟢" "🟢" "🟥" "🟢" "🟢" "🟢" "🟥"
    md"`nonlinear_pem`"     "🟢" "🟢" "🔶" "🟢" "🟥" "🟢" "🟥" "🟥"
    md"`arx`"               "🟥" "🟢" "🟥" "🔶" "🟢" "🟢" "🟥" "🟢"
    md"`arxar`"             "🟥" "🟢" "🟢" "🟥" "🟢" "🟢" "🟥" "🟥"
    md"`plr`"               "🟥" "🟢" "🟢" "🟥" "🟢" "🟢" "🟥" "🔶"
    md"`era/okid`"          "🟢" "🟢" "🟥" "🟥" "🟥" "🟢" "🟥" "🟢"
    md"`impulseest`"        "🟥" "🟢" "🟥" "🟥" "🟢" "🟢" "🟥" "🟥"
    md"`tfest`"             "🟥" "🟥" "🟢" "🟥" "🟢" "🟢" "🟢" "🟥"
]

io = IOBuffer()
tab = pretty_table(io, data; header, tf=tf_html_default)
tab_algos = String(take!(io)) |> HTML
```
```@example ALGOVERVIEW
tab_algos # hide
```

#### Comments
- All methods can estimate **SISO** systems, i.e., systems with a single input and a single output.
- Missing from the comparison is whether an algorithm estimates a **transfer function or a statespace system**, this is because one can without loss convert one to the other by simply calling `tf/ss`. One notable exception is for non-causal transfer functions which cannot be represented as statespace systems, but those do not appear very often.
- Several methods are listed as 🟥 on **nonlinearities**, but it is oftentimes possible to handle known input nonlinearities by adding nonlinearly transformed versions of the input to the dataset. Known output nonlinearities that are invertible can be handled by similarly applying the inverse nonlinearity to the data before estimation. Only [`newpem`](@ref) has explicit methods for estimating parameters of nonlinearities. [`arx`](@ref) is listed as 🔶, since with the correct `estimator` option that promotes sparsity, it is possible to find the most appropriate nonlinearity among a set of candidates. However, no explicit support for this is provided.
- **Custom loss functions** are sometimes supported explicitly, such as for [`newpem`](@ref), but often supported by providing a custom `estimator` for methods that solve a problem on the form ``\operatorname{argmin}_w \sum_i \operatorname{loss}(e_i) \quad \forall e_i \in \{e = y - Aw\}``. The default estimator in these cases is always `\`, i.e., to solve a least-squares problem. Useful alternatives are, e.g., [`TotalLeastSquares.tls`](https://github.com/baggepinnen/TotalLeastSquares.jl) and `TotalLeastSquares.irls`. This can be useful to increase robustness w.r.t. noise etc.
- In specific situations it is possible to use any method with **multiple datasets** by simply concatenating two datasets like `[d1 d2]`. This is only recommended if the state of the system in the end of the first dataset is very close to the state of the system in the beginning of the second dataset, for example, if all experiments start and end at rest in the origin.
- Some methods estimate explicit **disturbance models**, such as [`plr`](@ref) and [`arxar`](@ref), whereas other methods estimate observers with an *implicit* disturbance model, such as [`newpem`](@ref) and [`subspaceid`](@ref). All methods that estimate disturbance models are able to account for input disturbance (also referred to as dynamic disturbance or load disturbance). [`ControlSystemIdentification.nonlinear_pem`](@ref) is listed as 🔶 since it allows for the estimation of a disturbance model, but the user has to encode the model in the dynamics manually.

## Other resources
- [YouTube playlist](https://youtube.com/playlist?list=PLC0QOsNQS8ha6SwaNOZDXyG9Bj8MPbF-q&si=AiOZVhBVwYReDrAm) with tutorials using this package.
- For estimation of linear **time-varying** models (LTV), see [LTVModels.jl](https://github.com/baggepinnen/LTVModels.jl).
- For estimation of linear and nonlinear **grey-box models in continuous time**, see [DifferentialEquations.jl (parameter estimation)](https://docs.sciml.ai/DiffEqParamEstim/stable/)
- Estimation of **nonlinear black-box models in continuous time** [DiffEqFlux.jl](https://github.com/JuliaDiffEq/DiffEqFlux.jl/) and [DataDrivenDiffEq.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/)
- For more advanced **spectral estimation**, cross coherence, etc., see [LPVSpectral.jl](https://github.com/baggepinnen/LPVSpectral.jl)
- This package interacts well with [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl). See [example file](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/controlsystems.jl).
- **State estimation** is facilitated by [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl).
