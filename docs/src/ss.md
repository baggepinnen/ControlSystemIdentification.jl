# LTI state-space models

There exist two methods for identification of statespace models, [`n4sid`](@ref) and [`pem`](@ref). `n4sid` uses subspace-based identification whereas `pem` solves the prediction-error problem using an iterative optimization method (from Optim.jl). If unsure which method to use, try `n4sid` first.

## Subspace based identification using n4sid
```julia
d = iddata(y,u,sampletime)
sys = n4sid(d, :auto; verbose=false)
# or use a robust version of svd if y has outliers or missing values
using TotalLeastSquares
sys = n4sid(d, :auto; verbose=false, svd=x->rpca(x)[3])
```
Estimate a statespace model using the n4sid method. Returns an object of type [`N4SIDResult`](@ref) where the model is accessed as `sys.sys`.

#### Arguments:
- `d`: Identification data object, created using `iddata(y,u,sampletime)`.
- `y`: Measurements N×ny
- `u`: Control signal N×nu
- `r`: Rank of the model (model order)
- `verbose`: Print stuff?
- `Wf`: A frequency-domain model of measurement disturbances. To focus the attention of the model on a narrow frequency band, try something like `Wf = Bandstop(lower, upper, fs=1/Ts)` to indicate that there are disturbances *outside* this band.
- `i`: Algorithm parameter, generally no need to tune this
- `γ`: Set this to a value between (0,1) to stabilize unstable models such that the largest eigenvalue has magnitude γ.

The frequency weighting is borrowing ideas from
*"Frequency Weighted Subspace Based System Identification in the Frequency Domain", Tomas McKelvey 1996*. In particular, we apply the output frequency weight matrix (Fy) as it appears in eqs. (16)-(18).

### ERA and OKID
See
```julia
sys = era(d::AbstractIdData, r, m = 2r, n = 2r, l = 5r)
H   = okid(d::AbstractIdData, r, l = 5r)
```
[`era`](@ref), [`okid`](@ref).

### Filtering and simulation
Models can be simulated using `lsim` from ControlSystems.jl and using [`simulate`](@ref). You may also convert the model to a [`KalmanFilter`](@ref) from [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl) by calling `KalmanFilter(sys)`, after which you can perform filtering and smoothing etc. with the utilities provided for a `KalmanFilter`.




## PEM
A simple algorithm for identification of discrete-time LTI systems on state-space form:
```math
x' = Ax + Bu + Ke
y  = Cx + e
```
is provided. The user can choose to minimize either prediction errors or simulation errors, with arbitrary metrics, i.e., not limited to squared errors.

The result of the identification is a custom type `StateSpaceNoise <: ControlSystems.LTISystem`, with fields `A,B,K`, representing the dynamics matrix, input matrix and Kalman gain matrix, respectively. The observation matrix `C` is not stored, as this is always given by `[I 0]` (you can still access it through `sys.C` thanks to `getproperty`).

This package also supports estimating models on the form
```math
Ay = Bu + Cw
```
through pseudo-linear regression or
```math
Ay = Bu + 1/D w
```
through the generalized least squares method ([`arxar`](@ref)).
Estimation of the more general model form
```math
Ay = B/F u + C/D w
```
or any of its other special cases is not supported. Since those models are also LTI systems, estimating a state-space model is in some sense equivalent.

Transfer-function estimation through spectral methods is supported through the functions [`tfest`](@ref) and [`coherence`](@ref).

### Usage example
Below, we generate a system and simulate it forward in time. We then try to estimate a model based on the input and output sequences.
```julia
using ControlSystemIdentification, ControlSystems, Random, LinearAlgebra

function ⟂(x)
    u,s,v = svd(x)
    u*v
end
function generate_system(nx,ny,nu)
    U,S  = ⟂(randn(nx,nx)), diagm(0=>0.2 .+ 0.5rand(nx))
    A    = S*U
    B   = randn(nx,nu)
    C   = randn(ny,nx)
    sys = ss(A,B,C,0,1)
end

Random.seed!(1)
T   = 1000                      # Number of time steps
nx  = 3                         # Number of poles in the true system
nu  = 1                         # Number of control inputs
ny  = 1                         # Number of outputs
x0  = randn(nx)                 # Initial state
sim(sys,u,x0=x0) = lsim(sys, u', 1:T, x0=x0)[1]' # Helper function
sys = generate_system(nx,nu,ny)
u   = randn(nu,T)               # Generate random input
y   = sim(sys, u, x0)           # Simulate system
d   = iddata(y,u,1)

sysh,x0h,opt = pem(d, nx=nx, focus=:prediction) # Estimate model

yh = predict(sysh, d, x0h)      # Predict using estimated model
plot([y; yh]', lab=["y" "ŷ"])   # Plot prediction and true output
```

We can also simulate the system with colored noise, necessitating estimating also noise models.
```julia
σu = 0.1 # Noise variances
σy = 0.1

sysn = generate_system(nx,nu,ny)             # Noise system
un   = u + sim(sysn, σu*randn(size(u)),0*x0) # Input + load disturbance
y    = sim(sys, un, x0)
yn   = y + sim(sysn, σy*randn(size(u)),0*x0) # Output + measurement noise
dn   = iddata(yn,un,1)
```
The system now has `3nx` poles, `nx` for the system dynamics, and `nx` for each noise model, we indicated this to the main estimation function `pem`:
```julia
sysh,x0h,opt = pem(dn,nx=3nx, focus=:prediction)
yh           = predict(sysh, dn, x0h) # Form prediction
plot([y; yh]', lab=["y" "ŷ"])             # Compare true output (without noise) to prediction
```

We can have a look at the singular values of a balanced system Gramian:
```julia
s    = ss(sysh)   # Convert to standard state-space type
s2,G = balreal(s) # Form balanced representation (obs. and ctrb. Gramians are the same
diag(G)           # Singular values of Gramians

# 9-element Array{Float64,1}:
#  3.5972307807882844
#  0.19777167699663994
#  0.0622528285731599
#  0.004322765397504325
#  0.004270259700592557
#  0.003243449461350837
#  0.003150873301312319
#  0.0005827927965893053
#  0.00029732262107216666
```
Note that there are 3 big singular values, corresponding to the system poles, there are also 2×3 smaller singular values, corresponding to the noise dynamics.

The estimated noise model can be extracted by `noise_model(sys)`, we can visualize it with a bodeplot.
```julia
bodeplot(noise_model(sysh), exp10.(range(-3, stop=0, length=200)), title="Estimated noise dynamics")
```
See the [example notebooks](
https://github.com/JuliaControl/ControlExamples.jl?files=1) for these plots.

### Call signature
`sys, x0, opt = pem(d; nx, kwargs...)`
#### Arguments:
- `d`: Identification data object, created using `iddata(y, u sampletime)`
    - `y`: Measurements, either a matrix with time along dim 2, or a vector of vectors
    - `u`: Control signals, same structure as `y`
- `nx`: Number of poles in the estimated system. This number should be chosen as number of system poles plus number of poles in noise models for measurement noise and load disturbances.
- `focus`: Either `:prediction` or `:simulation`. If `:simulation` is chosen, a two stage problem is solved with prediction focus first, followed by a refinement for simulation focus.
- `metric`: A Function determining how the size of the residuals is measured, default `sse` (e'e), but any Function such as `norm`, `e->sum(abs,e)` or `e -> e'Q*e` could be used.
- `regularizer(p) = 0`: function for regularization of the parameter vector `p`. The structure of `p` is detailed below. L₂ regularization, for instance, can be achieved by `regularizer = p->sum(abs2, p)`
- `solver` Defaults to `Optim.BFGS()`
- `kwargs`: additional keyword arguments are sent to [`Optim.Options`](http://julianlsolvers.github.io/Optim.jl/stable/#user/config/).

#### Structure of parameter vector `p`
The parameter vector is of type [`ComponentVector`](https://github.com/jonniedie/ComponentArrays.jl) and the fields `A,B,K,x0` can be accessed as `p.A` etc. The internal storage is according to
```julia
A  = size(nx,nx)
B  = size(nx,nu)
K  = size(nx,ny)
x0 = size(nx)
p  = [A[:];B[:];K[:];x0]
```

### Return values
- `sys::StateSpaceNoise`: identified system. Can be converted to `StateSpace` by `convert(StateSpace, sys)` or `ss(sys)`, but this will discard the Kalman gain matrix, see `innovation_form` to obtain a predictor system.
- `x0`: Estimated initial state
- `opt`: Optimization problem structure. Contains info of the result of the optimization problem

### Functions
- `pem`: Main estimation function, see above.
- `predict(sys, d, x0=zeros)`: Form predictions using estimated `sys`, this essentially runs a stationary Kalman filter.
- `simulate(sys, u, x0=zeros)`: Simulate the system using input `u`. The noise model and Kalman gain does not have any influence on the simulated output.
- `innovation_form`: Extract the predictor model from the estimated system (`ss(A-KC,[B K],C,D)`).
- `noise_model`: Extract the noise model from the estimated system (`I - innovation_form(sys)`).


### Internals
Internally, [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) is used to optimize the system parameters, using automatic differentiation to calculate gradients (and Hessians where applicable). Optim solver options can be controlled by passing keyword arguments to `pem`, and by passing a manually constructed solver object. The default solver is [`BFGS()`](http://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/)


