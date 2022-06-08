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

If you are new to this package, start your journey through the documentation by learning about [Identification data](@ref). Examples are provided in the [Examples](@ref) section and in the form of jupyter notebooks [here](
https://github.com/JuliaControl/ControlExamples.jl). An introductory video is available below (system identification example starts around 55 minutes)

```@raw html
<iframe width="560" height="315" src="https://www.youtube.com/embed/Fdz2Fsm1aTY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```




# Other resources
- For estimation of linear **time-varying** models (LTV), see [LTVModels.jl](https://github.com/baggepinnen/LTVModels.jl).
- For estimation of linear and nonlinear **grey-box models in continuous time**, see [DifferentialEquations.jl (parameter estimation)](http://docs.juliadiffeq.org/stable/analysis/parameter_estimation.html)
- Estimation of **nonlinear black-box models in continuous time** [DiffEqFlux.jl](https://github.com/JuliaDiffEq/DiffEqFlux.jl/) and in discrete time [Flux.jl](https://github.com/FluxML/Flux.jl)
- For more advanced **spectral estimation**, cross coherence, etc., see [LPVSpectral.jl](https://github.com/baggepinnen/LPVSpectral.jl)
- This package interacts well with [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl). See [example file](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/controlsystems.jl).
- **State estimation** is facilitated by [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl).
