# ControlSystemIdentification

[![CI](https://github.com/baggepinnen/ControlSystemIdentification.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/ControlSystemIdentification.jl/actions)
[![codecov](https://codecov.io/gh/baggepinnen/ControlSystemIdentification.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/ControlSystemIdentification.jl)
[![Documentation, stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/ControlSystemIdentification.jl/stable)
[![Documentation, latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/ControlSystemIdentification.jl/dev)

System identification for [ControlSystems.jl](https://github.com/JuliaControl/ControlSystems.jl/). Examples in the form of jupyter notebooks are provided [here](
https://github.com/JuliaControl/ControlExamples.jl?files=1).

See the [documentation](https://baggepinnen.github.io/ControlSystemIdentification.jl/stable) for help.


## Quick example:
```julia
using ControlSystemIdentification, ControlSystemsBase
Ts = 0.1
G  = c2d(DemoSystems.resonant(), Ts) # A true system to generate data from
u  = randn(1,1000)                   # A random input
y  = lsim(G,u).y                     # Simulated output
y .+= 0.01 .* randn.()               # add measurement noise
d  = iddata(y, u, Ts)                # package data in iddata object
sys = subspaceid(d, :auto)           # estimate state-space model using subspace-based identification
bodeplot([G, sys.sys], lab=["True" "" "n4sid" ""])
```