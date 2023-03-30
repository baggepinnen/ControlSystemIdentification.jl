# Transfer function estimation

This page documents how to estimate transfer functions, sometimes called ARX or ARMAX models, i.e. models on any of the forms
```math
\begin{aligned}
G(z) &= \dfrac{B(z)}{A(z)} = \dfrac{b_m z^m + \dots + b_0}{z^n + a_{n-1} z^{n-1} + \dots + a_0} \\
Ay &= Bu + w \\
Ay &= Bu + Cw \\
Ay &= Bu + 1/D w
\end{aligned}
```

This package estimates models in discrete time, but they may be converted to continuous-time models using the function [`d2c`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.d2c) from [ControlSystemsBase.jl](https://github.com/JuliaControl/ControlSystems.jl).

The methods available are:
## Functions
- [`arx`](@ref): Transfer-function estimation using least-squares fitting with a closed-form solution.
- [`arma`](@ref): Estimate an ARMA model (no control input).
- [`ar`](@ref): Estimate an AR model (no input).
- [`arma_ssa`](@ref) Estimate an ARMA model with estimated noise as input (no control input).
- [`plr`](@ref): Transfer-function estimation (ARMAX model) using pseudo-linear regression. [`armax`](@ref) is an alias for this function. This method estimates a noise model as well.
- [`arxar`](@ref): Transfer-function estimation using generalized least-squares method. This method estimates a noise model as well.
- [`getARXregressor`](@ref)/[`getARregressor`](@ref): For low-level control over the estimation
See docstrings for further help.

!!! note
    Most methods for estimation of transfer functions handle SISO, SIMO or MISO systems only. For estimation of MIMO systems, consider using state-space based methods and convert the result to a transfer function using `tf` after estimation if required. 

## Usage example:
```julia
N  = 2000     # Number of time steps
t  = 1:N
Δt = 1        # Sample time
u  = randn(N) # A random control input
G  = tf(0.8, [1,-0.9], 1)
y  = lsim(G,u,t)[1][:]
yn = y
d  = iddata(y,u,Δt)

na,nb = 1,1   # Number of polynomial coefficients

Gls = arx(d,na,nb,stochastic=false) # set stochastic to true to get a transfer function of MonteCarloMeasurements.Particles
@show Gls
# TransferFunction{ControlSystemsBase.SisoRational{Float64}}
#     0.8000000000000005
# --------------------------
# 1.0*z - 0.8999999999999997
```
As we can see, the model is perfectly recovered. In reality, the measurement signal is often affected by noise, in which case the estimation will suffer. To combat this, a few different options exist:
```julia
e  = randn(N)
yn = y + e    # Measurement signal with noise
d  = iddata(yn,u,Δt)

na,nb,nc = 1,1,1

Gls      = arx(d,na,nb, stochastic=true)     # Regular least-squares estimation
Gtls     = arx(d,na,nb, estimator=tls)       # Total least-squares estimation
Gwtls    = arx(d,na,nb, estimator=wtls_estimator(y,na,nb)) # Weighted Total least-squares estimation
Gplr, Gn = plr(d,na,nb,nc, initial_order=20) # Pseudo-linear regression
@show Gls; @show  Gtls; @show  Gwtls; @show  Gplr; @show  Gn;
# TransferFunction{ControlSystemsBase.SisoRational{MonteCarloMeasurements.Particles{Float64,500}}}
#     0.824 ± 0.029
# ---------------------
# 1.0*z - 0.713 ± 0.013

# Gtls = TransferFunction{ControlSystemsBase.SisoRational{Float64}}
#     1.848908051191616
# -------------------------
# 1.0*z - 0.774385918070221

# Gwtls = TransferFunction{ControlSystemsBase.SisoRational{Float64}}
#    0.8180228878106678
# -------------------------
# 1.0*z - 0.891939152690534

# Gplr = TransferFunction{ControlSystemsBase.SisoRational{Float64}}
#     0.8221837077656046
# --------------------------
# 1.0*z - 0.8896345125395438

# Gn = TransferFunction{ControlSystemsBase.SisoRational{Float64}}
#     0.9347035105826179
# --------------------------
# 1.0*z - 0.8896345125395438
```
We now see that the estimate using standard least-squares is heavily biased and it is wrongly certain about the estimate (notice the ± in the transfer function coefficients). Regular Total least-squares does not work well in this example, since not all variables in the regressor contain equally much noise. Weighted total least-squares does a reasonable job at recovering the true model. Pseudo-linear regression also fares okay, while simultaneously estimating a noise model. The helper function [`wtls_estimator`](@ref)`(y,na,nb)` returns a function that performs `wtls` using appropriately sized covariance matrices, based on the length of `y` and the model orders. Weighted total least-squares estimation is provided by [TotalLeastSquares.jl](https://github.com/baggepinnen/TotalLeastSquares.jl). See the [example notebooks](
https://github.com/JuliaControl/ControlExamples.jl?files=1) for more details.

Uncertain transfer function with `Particles` coefficients can be used like any other model. Try, e.g., `nyquistplot(Gls)` to get a Nyquist plot with confidence bounds.

See also function [`arma`](@ref) for estimation of signal models without inputs.

## Time-series modeling
Time-series modeling can be seen as special cases of transfer-function modeling where there are no control inputs. This package is primarily focused on control system identification, but we nevertheless provide two methods aimed at time-series estimation:
- [`ar`](@ref): Estimate an AR model (no input).
- [`arma_ssa`](@ref) Estimate an ARMA model with estimated noise as input (no control input).


## Code generation
To generate C-code for, e.g., simulating a system or filtering through an estimated transfer function, see [SymbolicControlSystems.jl](https://github.com/JuliaControl/SymbolicControlSystems.jl).

## Transfer-function API

```@docs
ControlSystemIdentification.arx
ControlSystemIdentification.ar
ControlSystemIdentification.arma
ControlSystemIdentification.plr
ControlSystemIdentification.arxar
ControlSystemIdentification.getARXregressor
ControlSystemIdentification.getARregressor
```