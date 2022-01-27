"""
    getARXregressor(y::AbstractVector,u::AbstractVecOrMat, na, nb; inputdelay = ones(Int, size(nb)))
Returns a shortened output signal `y` and a regressor matrix `A` such that the least-squares ARX model estimate of order `na,nb` is `y\\A`
Return a regressor matrix used to fit an ARX model on, e.g., the form
`A(z)y = B(z)f(u)`
with output `y` and input `u` where the order of autoregression is `na`,
the order of input moving average is `nb` and an optional input delay `inputdelay`. Caution, changing the input delay changes the order to `nb + inputdelay - 1`. An `inputdelay = 0` results in a direct term. 
# Example
Here we test the model with the Function `f(u) = √(|u|)`
```julia
A     = [1,2*0.7*1,1] # A(z) coeffs
B     = [10,5] # B(z) coeffs
u     = randn(100) # Simulate 100 time steps with Gaussian input
y     = filt(B,A,u)
yr,A  = getARXregressor(y,u,3,2) # We assume that we know the system order 3,2
x     = A\\yr # Estimate model polynomials
plot([yr A*x], lab=["Signal" "Prediction"])
```
For nonlinear ARX-models, see [BasisFunctionExpansions.jl](https://github.com/baggepinnen/BasisFunctionExpansions.jl/). See also `arx`
"""
function getARXregressor(y::AbstractVector, u::AbstractVecOrMat, na, nb; inputdelay = ones(Int, size(nb)))
    length(nb) == size(u, 2) ||
        throw(ArgumentError("Length of nb must equal number of input signals"))
    size(nb) == size(inputdelay) || throw(ArgumentError("inputdelay has to have the same structure as nb"))
    m = max(na, maximum(nb .+ inputdelay .- 1)) + 1 # Start of yr
    @assert m >= 1
    n = length(y) - m + 1 # Final length of yr
    @assert n <= length(y)
    A = toeplitz(y[m:m+n-1], y[m:-1:m-na])
    @assert size(A, 2) == na + 1
    y = A[:, 1] # extract yr
    A = A[:, 2:end]
    for i = 1:length(nb)
        nb[i] <= 0 && continue
        s = m - inputdelay[i]
        A = [A toeplitz(u[s:s+n-1, i], u[s:-1:s-(nb[i])+1, i])]
    end
    return y, A
end
getARXregressor(d::AbstractIdData, na, nb; inputdelay = ones(Int, size(nb))) =
    getARXregressor(time1(output(d)), time1(input(d)), na, nb; inputdelay = inputdelay)

function getARregressor(dy::AbstractIdData, na)
    noutputs(dy) == 1 || throw(ArgumentError("Only 1d time series supported"))
    y = time1(output(dy))
    getARregressor(vec(y), na)
end

"""
    yt,A = getARregressor(y::AbstractVector, na)

Returns values such that `x = A\\yt`. See [`getARXregressor`](@ref) for more details.
"""
function getARregressor(y::AbstractVector, na)
    m = na + 1 # Start of yr
    n = length(y) - m + 1 # Final length of yr
    A = toeplitz(y[m:m+n-1], y[m:-1:1])
    @assert size(A, 2) == na + 1
    y = A[:, 1] # extract yr
    A = A[:, 2:end]
    return y, A
end




"""
    Gtf = arx(d::AbstractIdData, na, nb; inputdelay = ones(Int, size(nb)), λ = 0, estimator=\\, stochastic=false)

Fit a transfer Function to data using an ARX model and equation error minimization.

- `nb` and `na` are the number of coefficients of the numerator and denominator polynomials.
Input delay can be added via `inputdelay = d`, which corresponds to an additional delay of `z^-d`.
An `inputdelay = 0` results in a direct term.
The highest order of the B polynomial is given by `nb + inputdelay - 1`.  `λ > 0` can be provided for L₂ regularization.
`estimator` defaults to \\ (least squares), alternatives are `estimator = tls` for total least-squares estimation. 
`arx(Δt,yn,u,na,nb, estimator=wtls_estimator(y,na,nb)` is potentially more robust in the presence of
heavy measurement noise. The number of free parameters is `na+nb` 
- `stochastic`: if true, returns a transfer function with uncertain parameters represented by `MonteCarloMeasurements.Particles`.

Supports MISO estimation by supplying an iddata with a matrix `u`, with nb = [nb₁, nb₂...] and optional inputdelay = [d₁, d₂...]
"""
function arx(d::AbstractIdData, na, nb; inputdelay = ones(Int, size(nb)), λ = 0, estimator = \, stochastic = false)
    y, u, h = time1(output(d)), time1(input(d)), sampletime(d)
    @assert size(y, 2) == 1 "arx only supports single output."
    # all(nb .<= na) || throw(DomainError(nb,"nb must be <= na"))
    na >= 0 || throw(ArgumentError("na must be positive"))
    size(nb) == size(inputdelay) || throw(ArgumentError("inputdelay has to have the same structure as nb"))
    y_train, A = getARXregressor(vec(y), u, na, nb; inputdelay)
    w = ls(A, y_train, λ, estimator)
    a, b = params2poly2(w, na, nb; inputdelay)
    model = tf(b, a, h)
    if stochastic
        local Σ
        try
            Σ = parameter_covariance(y_train, A, w, λ)
        catch
            return minreal(model)
        end
        return minreal(TransferFunction(Particles, model, Σ))
    end

    return minreal(model)
end

"""
    residuals(ARX::TransferFunction, d::InputOutputData)

Calculates the residuals `v = Ay - Bu` of an ARX process and InputOutputData d. The length of the returned residuals is `length(d) - max(na, nb)`
# Example:
```jldoctest
julia> ARX = tf(1, [1, -1], 1)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Int64}}
  1
-----
z - 1

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> u = 1:5
1:5

julia> y = lsim(ARX, u, 1:5)[1][:]
5-element Vector{Float64}:
  0.0
  1.0
  3.0
  6.0
 10.0

julia> d = iddata(y, u)
InputOutput data of length 5 with 1 outputs and 1 inputs

julia> residuals(ARX, d)
4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
```
"""
function residuals(ARX::TransferFunction, d::InputOutputData)
    size(ARX, 2) == ninputs(d) || throw(DomainError(d, "number of inputs $(ninputs(d)) does not match ARX model (expects $(size(ARX, 2)) inputs)"))

    w, a, b, inputdelay = params(ARX)
    na = length(a)
    nb = map(length, vec(b))
    
    y, A = getARXregressor(d, na, nb, inputdelay = inputdelay)
    ypred = A * w
    v = y .- ypred
    return v
end

"""
    predict(ARX::TransferFunction, d::InputOutputData)

One step ahead prediction for an ARX process.  The length of the returned prediction is `length(d) - max(na, nb)`
# Example:
```jldoctest
julia> predict(tf(1, [1, -1], 1), iddata(1:10, 1:10))
9-element Vector{Int64}:
  2
  4
  6
  8
 10
 12
 14
 16
 18
```
"""
function predict(ARX::TransferFunction, d::InputOutputData)
    size(ARX, 2) == ninputs(d) || throw(DomainError(d, "number of inputs $(ninputs(d)) does not match ARX model (expects $(size(ARX, 2)) inputs)"))

    w, a, b, inputdelay = params(ARX)
    na = length(a)
    nb = map(length, vec(b))
    
    y, A = getARXregressor(d, na, nb, inputdelay = inputdelay)
    ypred = A * w
    return ypred
end

"""
    ar(d::AbstractIdData, na; λ=0, estimator=\\, scaleB=false, stochastic=false)

Estimate an AR transfer function `G = 1/A`, the AR process is defined as `A(z⁻¹)y(t) = e(t)`

# Arguments:
- `d`: IdData, see [`iddata`](@ref)
- `na`: order of the model
- `λ`: `λ > 0` can be provided for L₂ regularization
- `estimator`: e.g. `\\,tls,irls,rtls`
- `scaleB`: Whether or not to scale the numerator using the variance of the prediction error.
- `stochastic`: if true, returns a transfer function with uncertain parameters represented by `MonteCarloMeasurements.Particles`.

Estimation of AR models using least-squares is known to struggle with heavy measurement noise, using `estimator = tls` can improve the result in this case.

# Example
```jldoctest
julia> N = 10000
10000

julia> e = [-0.2; zeros(N-1)] # noise e
10000-element Vector{Float64}:
[...]

julia> G = tf([1, 0], [1, -0.9], 1) # AR transfer function
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
   1.0z
----------
1.0z - 0.9

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> y = lsim(G, e, 1:N)[1][:] # Get output of AR transfer function from input noise e
10000-element Vector{Float64}:
[...]

julia> Gest = ar(iddata(y), 1) # Estimate AR transfer function from output y
TransferFunction{Discrete{Float64}, ControlSystems.SisoRational{Float64}}
          1.0z
-------------------------
1.0z - 0.8999999999999998

Sample Time: 1.0 (seconds)
Discrete-time transfer function model

julia> G ≈ Gest # Test if estimation was correct
true

julia> eest = lsim(1/Gest, y, 1:N)[1][:] # recover the input noise e from output y and estimated transfer function Gest
10000-element Vector{Float64}:
[...]

julia> isapprox(eest, e, atol = eps()) # input noise correct recovered
true 
```
"""
function ar(d::AbstractIdData, na; λ = 0, estimator = \, scaleB = false, stochastic = false)
    noutputs(d) == 1 || throw(ArgumentError("Only 1d time series supported"))
    y = vec(time1(output(d)))
    na >= 1 || throw(ArgumentError("na must be positive"))
    y_train, A = getARregressor(y, na)
    w = ls(A, y_train, λ, estimator)
    a, b = params2poly(w, na)
    if scaleB
        b = √mean(abs2, y_train - A * w) * b
    end
    model = tf(b, vec(a), sampletime(d))
    if stochastic
        Σ = parameter_covariance(y_train, A, w, λ)
        return TransferFunction(Particles, model, Σ)
    end
    model
end

"""
    G, H, e = arxar(d::InputOutputData, na::Int, nb::Union{Int, Vector{Int}}, nd::Int)

Estimate the ARXAR model `Ay = Bu + v`, where `v = He` and `H = 1/D`, using generalized least-squares method. For more information see Söderström - Convergence properties of the generalized least squares identification method, 1974. 

# Arguments:
- `d`: iddata
- `na`: order of A
- `nb`: number of coefficients in B, the order is determined by `nb + inputdelay - 1`. In MISO estimation it takes the form nb = [nb₁, nb₂...]. 
- `nd`: order of D

# Keyword Arguments:
- `H = nothing`: prior knowledge about the AR noise model
- `inputdelay = ones(Int, size(nb))`: optional delay of input, inputdelay = 0 results in a direct term, takes the form inputdelay = [d₁, d₂...] in MISO estimation 
- `λ = 0`: `λ > 0` can be provided for L₂ regularization
- `estimator = \\`: e.g. `\\,tls,irls,rtls`, the latter three require `using TotalLeastSquares`
- `δmin = 10e-4`: Minimal change in the power of e, that specifies convergence.
- `iterations = 10`: maximum number of iterations.
- `verbose = false`: if true, more information is printed

# Example:
```
julia> N = 500 
500

julia> sim(G, u) = lsim(G, u, 1:N)[1][:]
sim (generic function with 1 method)

julia> A = tf([1, -0.8], [1, 0], 1)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
1.0z - 0.8
----------
   1.0z

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> B = tf([0, 1], [1, 0], 1)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Int64}}
1
-
z

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> G = minreal(B / A)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
   1.0
----------
1.0z - 0.8

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> D = tf([1, 0.7], [1, 0], 1)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
1.0z + 0.7
----------
   1.0z

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> H = 1 / D
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
   1.0z
----------
1.0z + 0.7

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> u, e = randn(1, N), randn(1, N)
[...]

julia> y, v = sim(G, u), sim(H * (1/A), e) # simulate process
[...]

julia> d = iddata(y .+ v, u, 1)
InputOutput data of length 500 with 1 outputs and 1 inputs

julia> na, nb , nd = 1, 1, 1
(1, 1, 1)

julia> Gest, Hest, res = arxar(d, na, nb, nd)
(G = TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
   0.9987917259291642
-------------------------
1.0z - 0.7937837464682017

Sample Time: 1 (seconds)
Discrete-time transfer function model, H = TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
          1.0z
-------------------------
1.0z + 0.7019519225937721

Sample Time: 1 (seconds)
Discrete-time transfer function model, e = [...]
```
"""
function arxar(d::InputOutputData, na::Int, nb::Union{Int, AbstractVector{Int}}, nd::Int;
    H::Union{TransferFunction, Nothing} = nothing,
    inputdelay      = ones(Int, size(nb)),
    δmin::Real      = 0.001,
    iterations::Int = 10,
    estimator       = \,
    verbose::Bool   = false,
    λ::Real         = 0
)
    # Input Checking
    na >= 0 || throw(ArgumentError("na($na) must be positive"))
    all(nb .>= 0 )|| throw(ArgumentError("nb($nb) must be positive"))
    nd >= 1 || throw(ArgumentError("nd($nd) must be positive"))
    iterations >= 1 || throw(DomainError("iterations($iterations) must be >0"))
	δmin > 0 || throw(ArgumentError("δmin($δmin) must be positive"))
    ninputs(d) == length(nb) || throw(ArgumentError("Length of nb ($(length(nb))) must equal number of input signals ($(ninputs(d)))"))

    Ts = sampletime(d)

    # 1. initialize H, GF and v to bring them to scope
    if H === nothing
        H = tf([0], [1], Ts)
        iter = 0 # initialization
    else
        iter = 1 # initial noisemodel is known, H is present
    end
    GF = tf([0], [1], Ts)
    v  = 0

    # Iterate
    eOld    = 0
    δ       = δmin + 1
    timeVec = timevec(d)
    sim(G,u) = lsim(G, u, timeVec)[1]
    while iter <= iterations && δ >= δmin
        # Filter input/output according to errormodel H, after initialization
        if iter > 0
            yF = sim(1/H, output(d))
            if ninputs(d) == 1
                uF = sim(1/H, input(d))
            else
                uF = fill(1.0, size(d.u)) 
                for i in 1:ninputs(d)
                    uF[i, :] = sim(1/H, d.u[i:i,:])
                end
            end
        else
            # pass unfiltered signal for initialization
            yF = copy(output(d))
            uF = copy(input(d))
        end
		dF = iddata(yF, uF, d.Ts)

		# 2. fit arx model
        GF = arx(dF, na, nb; estimator, λ, inputdelay)

        # 3. Evaluate residuals
        v = residuals(GF, d)

        # 4. check if converged
        e = var(v)
        if eOld != 0
            δ = abs((eOld - e) / e)
        end
        verbose && println("iter: $iter, δ: $δ, e: $e")
        eOld = e

        # 5. estimate new noise model from residuals
        dH = iddata(v, d.Ts)
        H = ar(dH, nd; estimator)
        
        iter += 1
    end

    # total residuals e
    e = lsim(1/H, v', timevec(v, Ts))[1][:]
    return (G = GF, H, e)
end


"""
    arxar_predictor(G, H)

Convert the models obtained from `arxar` into a `PredictionStateSpace`. Note that the predictor in this case will predict the sum of the system and noise output, while a simulation will predict the system output alone. 
# Examples:
```julia
Gp = ControlSystemIdentification.arxar_predictor(Gest, Hest) 
pe = ControlSystemIdentification.prediction_error(Gp)
pd = ControlSystemIdentification.predictiondata(d)
ε = lsim(pe, pd)[1] # estimate innovation sequence

yp = predict(Gp, d)  # prediction includes prediction of noise
ys = simulate(Gp, d) # simulation includes only system output
```
"""
function arxar_predictor(G, H)
    Ts = G.Ts
    A = denvec(G[1,1])[]
    H2 = ss(minreal(H*tf(1,[1,0],1))) # is good idea

    Ge = ss(G)
    Hs = ss(H2)
    # Hs = ss(tf(1, A, Ts)*H2) # not good idea
    A,B,C,D = ssdata(Ge)
    Ad,Bd,Cd,Dd = ssdata(Hs)
    Ae = ControlSystems.blockdiag(A, Ad)
    Ae[1:Hs.ny, Ge.nx+1:end] .= Cd
    Be = [B; 0Bd]
    Ce = [C 0Cd]
    De = D
    syse = ss(Ae,Be,Ce,De,Ts)
    Bw = [0B; Bd]
    Rw = Bw*I(1)*Bw'
    # Rw = I(syse.nx)
    Re = Matrix(0.0001I(syse.ny))
    K = kalman(syse, Rw, Re)
    PredictionStateSpace(syse, K, Rw, Re)
end



function reversal_ls(A, y)
    n = size(A, 2)
    J = I(n)[:, end:-1:1]
    R = A'A
    R .= 1 / 2 .* (R + J * R' * J)
    R \ (A'y)
end


function ls(A, y, λ = 0, estimator = \)
    if λ == 0
        w = estimator(A, y)
    else
        w = estimator([A; λ * I], [y; zeros(size(A, 2))])
    end
    w
end

"""
    G, Gn = plr(d::AbstractIdData,na,nb,nc; initial_order = 20)

Perform pseudo-linear regression to estimate a model on the form
`Ay = Bu + Cw`
The residual sequence is estimated by first estimating a high-order arx model, whereafter the estimated residual sequence is included in a second estimation problem. The return values are the estimated system model, and the estimated noise model. `G` and `Gn` will always have the same denominator polynomial.

`armax` is an alias for `plr`. See also [`pem`](@ref), [`ar`](@ref), [`arx`](@ref)
"""
function plr(d::AbstractIdData, na, nb, nc; initial_order = 20, method = :ls)
    y, u, h = time1(output(d)), time1(input(d)), sampletime(d)
    all(nb .<= na) || throw(DomainError(nb, "nb must be <= na"))
    na >= 1 || throw(ArgumentError("na must be positive"))
    # na -= 1
    y_train, A = getARXregressor(y, u, initial_order, initial_order)
    w1 = A \ y_train
    yhat = A * w1
    ehat = yhat - y_train
    ΔN = length(y) - length(ehat)
    y_train, A =
        getARXregressor(y[ΔN+1:end-1], [u[ΔN+1:end-1, :] ehat[1:end-1]], na, [nb; nc])
    if method == :wtls
        w = wtls_estimator(y[ΔN+1:end-1], na, nb + nc, [zeros(nb); ones(nc)])(A, y_train)
    elseif method == :rtls
        w = rtls(A, y_train)
    elseif method == :ls
        w = A \ y_train
    elseif method isa Function
        w = method(A, y_train)
    end
    a, b = params2poly(w, na, nb)
    model = tf(b, a, h)
    c = w[na+sum(nb)+1:end]
    noise_model = tf(c, a, h)
    model, noise_model
end

const armax = plr


"""
    model = arma(d::AbstractIdData, na, nc; initial_order=20, method=:ls)

Estimate a Autoregressive Moving Average model with `na` coefficients in the denominator and `nc` coefficients in the numerator.
Returns the model and the estimated noise sequence driving the system.

# Arguments:
- `d`: iddata
- `initial_order`: An initial AR model of this order is used to estimate the residuals
- `estimator`: A function `(A,y)->minimizeₓ(Ax-y)` default is `\` but another option is `wtls_estimator(1:length(y)-initial_order,na,nc,ones(nc))`

See also [`estimate_residuals`](@ref)
"""
function arma(d::AbstractIdData, na, nc; initial_order = 20, estimator = \)
    y, h = time1(output(d)), sampletime(d)
    all(nc .<= na) || throw(DomainError(nb, "nc must be <= na"))
    na >= 1 || throw(ArgumentError("na must be positive"))
    # na -= 1
    y_train, A = getARregressor(y, initial_order)
    # w1 = method isa Function ? method(A,y_train) : tls(A,y_train)
    w1 = A \ y_train # The above seems like it should work but tests indicate that it makes performance worse, maybe exactly because it removes too much correlation in the residuals
    yhat = A * vec(w1)
    ehat = y_train - yhat
    σ = √mean(abs2, ehat)
    ΔN = length(y) - length(ehat)
    y2 = @view y[ΔN:end-1]
    y_train, A = getARXregressor(y2, ehat, na, nc)
    w = estimator(A, y_train)
    a, b = params2poly(w, na, nc)
    # b[1] = 1
    b = σ * b
    model = tf(b, a, h)
    model
end

"""
Freqresp
"""
function vconv(w::T, vars::AbstractArray{R}) where {T, R}
    n = size(vars,1)
    S = promote_type(T, R)
    c = zero(S)
    vars = reverse(vars)
    sv = [1,0,-1,0]
    for i in 1:n, j in 1:n
        s = sv[abs(i-j) % 4 + 1]
        val = (w^(i-1) * w^(j-1) * vars[i]*vars[j])[]
        @inbounds c += s*val
    end
    return c
end


"""
    tfest(
        data::FRD,
        p0,
        link = log ∘ abs;
        freq_weight = sqrt(data.w[1]*data.w[end]),
        refine = true,
        opt = BFGS(),
        opts = Optim.Options(
            store_trace       = true,
            show_trace        = true,
            show_every        = 1,
            iterations        = 100,
            allow_f_increases = false,
            time_limit        = 100,
            x_tol             = 0,
            f_tol             = 0,
            g_tol             = 1e-8,
            f_calls_limit     = 0,
            g_calls_limit     = 0,
        ),
    )

Fit a parametric transfer function to frequency-domain data.

The initial pahse of the optimization solves
```math
\\operatorname{minimize}_{B,A}{|| B - A|l|^2||}
```
and the second stage (if refine=true) solves 
```math
\\operatorname{minimize}_{B,A}{|| \\text{link}\\left(\\dfrac{B}{A}\\right) - \\text{link}\\left(l\\right)||}
```
(`abs2(link(B/A) - link(l))`)

# Arguments:
- `data`: An `FRD` onbject with frequency domain data.
- `p0`: Initial parameter guess. Can be a `NamedTuple` or `ComponentVector` with fields `b,a` specifying numerator and denominator as they appear in the call to `tf`, i.e., `(b = [1.0], a = [1.0,1.0,1.0])`. Can also be an instace of `TransferFunction`.
- `link`: By default, phase information is discarded in the fitting. To include phase, change to `link = log`.
- `freq_weight`: Apply weighting with the inverse frequency. The value determines the cutoff frequency before which the weight is constant, after which the weight decreases linearly. Defaults to the geometric mean of the smallest and largest frequency.
- `refine`: Indicate whether or not a second optimization stage is performed to refine the results of the first.
- `opt`: The Optim optimizer to use.
- `opts`: `Optim.Options` controlling the solver options.

See also [`minimum_phase`](@ref) to transform a possibly non-minimum phase system to minimum phase.
"""
function tfest(
    data::FRD,
    p0::Union{NamedTuple, ComponentArray},
    link = log ∘ abs;
    freq_weight = sqrt(data.w[2] * data.w[end]),
    refine = true,
    opt = BFGS(),
    opts = Optim.Options(
        store_trace       = true,
        show_trace        = true,
        show_every        = 5,
        iterations        = 10000,
        allow_f_increases = false,
        time_limit        = 100,
        x_tol             = 0,
        f_tol             = 0,
        g_tol             = 1e-8,
        f_calls_limit     = 0,
        g_calls_limit     = 0,
    ),
)
    # TODO: implement https://epubs.siam.org/doi/pdf/10.1137/140961511 which is the algorithm used in matlab, https://se.mathworks.com/help/ident/ref/tfest.html#bvgwvmc

    ladr = @. link(data.r)
    w, l = data.w, data.r
    if freq_weight > 0
        wv = 1 ./ (data.w .+ freq_weight)
    end

    function loss2(p)
        a, b = p.a, p.b
        G = tf(b, a)
        mag = vec(freqresp(G, data.w))
        @. mag = link(mag) - ladr
        if freq_weight > 0
            mag .*= wv
        end
        mean(abs2, mag)
    end

    function loss(p)
        a, b = p.a, p.b
        mean(eachindex(w)) do i
            abs2(vconv(w[i], b) - vconv(w[i], a) * abs2(l[i]))
        end
    end

    res = Optim.optimize(loss, ComponentVector(p0), opt, opts, autodiff = :forward)
    if refine
        res = Optim.optimize(loss2, res.minimizer, opt, opts, autodiff = :forward)
    end

    tf(res.minimizer.b, res.minimizer.a)
end

tfest(data, G::LTISystem, args...; kwargs...) = tfest(tfest(data)[1], G, args...; kwargs...)

function tfest(data::FRD, G::LTISystem, args...; kwargs...)
    ControlSystems.issiso(G) || throw(ArgumentError("Can only fit SISO model to FRD"))
    ControlSystems.isdiscrete(G) && throw(DomainError("Continuous-time model expected"))
    b, a = numvec(G)[], denvec(G)[]
    tfest(data, (; b = b, a = a), args...; kwargs...)
end

"""
    tfest(data::FRD, basis::AbstractStateSpace; 
        freq_weight = 1 ./ (data.w .+ data.w[2]),
        opt = BFGS(),
        metric::M = abs2,
        opts = Optim.Options(
            store_trace       = true,
            show_trace        = true,
            show_every        = 50,
            iterations        = 1000000,
            allow_f_increases = false,
            time_limit        = 100,
            x_tol             = 1e-5,
            f_tol             = 0,
            g_tol             = 1e-8,
            f_calls_limit     = 0,
            g_calls_limit     = 0,
    )

Fit a parametric transfer function to frequency-domain data using a pre-specified basis.

# Arguments:
- `data`: An `FRD` onbject with frequency domain data.
function kautz(a::AbstractVector)
- `basis`: A basis for the estimation. See, e.g., `laguerre, laguerre_oo, kautz`
- `freq_weight`: A vector of weights per frequency. The default is approximately `1/f`. 
- `opt`: The Optim optimizer to use.
- `opts`: `Optim.Options` controlling the solver options.
"""
function tfest(data::FRD, basis::AbstractStateSpace; 
    freq_weight = 1 ./ (data.w .+ data.w[2]),
    opt = BFGS(),
    metric::M = abs2,
    opts = Optim.Options(
        store_trace       = true,
        show_trace        = true,
        show_every        = 50,
        iterations        = 1000000,
        allow_f_increases = false,
        time_limit        = 100,
        x_tol             = 1e-5,
        f_tol             = 0,
        g_tol             = 1e-8,
        f_calls_limit     = 0,
        g_calls_limit     = 0,
    ),
) where M
    ω      = data.w
    Fs     = basis_responses(basis, ω, inverse=false)
    p0     = randn(length(Fs))
    resp_P = abs2.(data.r) .|> log
    Gmat   = reduce(hcat, Fs)
    Gθ     = similar(Gmat, size(Gmat, 1))

    function loss(p)
        mul!(Gθ, Gmat, p)
        c = zero(eltype(p))
        @inbounds for i in eachindex(ω)
            fp = log(abs2(Gθ[i]))
            c += metric(fp - resp_P[i]) * freq_weight[i]
        end
        c/length(ω)
    end
    
    res = Optim.optimize(
        loss,
        p0,
        opt,
        opts,
        # autodiff=:forward
    )
    F = sum_basis(basis, res.minimizer)
    if dcgain(F)[] < 0
        F = -F
        res.minimizer .= .- res.minimizer
    end
    F, res
end

"""
    arma_ssa(d::AbstractIdData, na, nc; L=nothing, estimator=\\, robust=false)

DOCSTRING

# Arguments:
- `d`: iddata
- `na`: number of denominator parameters
- `nc`: number of numerator parameters
- `L`: length of the lag-embedding used to separate signal and noise. `nothing` corresponds to automatic selection.
- `estimator`: The function to solve the least squares problem. Examples `\\,tls,irls,rtls`.
- `robust`: Use robust PCA to be resistant to outliers.
"""
function arma_ssa(d::AbstractIdData, na, nc; L = nothing, estimator = \, robust = false)
    y, h = time1(output(d)), sampletime(d)
    all(nc .<= na) || throw(DomainError(nc, "nc must be <= na"))
    na >= 1 || throw(ArgumentError("na must be positive"))

    L === nothing && (L = min(length(y) ÷ 2, 2000))
    H = hankel(y, L)
    if robust
        H, _, s = rpca(H)
        # yhat = unhankel(H)
    else
        s = svd(H)
    end
    yinds = 1:L-L÷4
    noiseinds = L-L÷4+1:L
    yhat = unhankel(s.U[:, yinds] * Diagonal(s.S[yinds]) * s.Vt[yinds, :])
    ehat = unhankel(s.U[:, noiseinds] * Diagonal(s.S[noiseinds]) * s.Vt[noiseinds, :])
    dhat = iddata(yhat, ehat, d.Ts)

    arma(dhat, na, nc, initial_order = L ÷ 5)
end


"""
    estimate_residuals(model, y)

Estimate the residuals driving the dynamics of an ARMA model.
"""
function estimate_residuals(model, yi)
    y = time1(output(yi))
    eest = zeros(length(y))
    na = length(denvec(model)[1, 1]) - 1
    nc = length(numvec(model)[1, 1])
    w = params(model)[1]
    for i = na:length(y)-1
        ϕ = [y[i:-1:i-na+1]; 0; eest[i-1:-1:i-nc+1]]
        yh = w'ϕ
        eest[i] = y[i+1] - yh
        ϕ[na+1] = eest[i]
        yh = w'ϕ
        eest[i] += y[i+1] - yh
    end
    oftype(output(yi), eest)
end



# Helper constructor to make a MISO system after MISO arx estimation
function ControlSystems.tf(
    b::AbstractVector{<:AbstractVector{<:Number}},
    a::AbstractVector{<:Number},
    h,
)
    tfs = map(b) do b
        tf(b, a, h)
    end
    hcat(tfs...)
end

"""
    wtls_estimator(y,na,nb, σu=0)

Create an estimator function for estimation of arx models in the presence of measurement noise. If the noise variance on the input `σu` (model errors) is known, this can be specified for increased accuracy.
"""
function wtls_estimator(y, na, nb, σu = 0)
    y = output(y)
    rowQ = [diagm(0 => [ones(na); σu .* ones(nb); 1]) for i = 1:length(y)-(na)]
    Qaa, Qay, Qyy = rowcovariance(rowQ)
    (A, y) -> wtls(A, y, Qaa, Qay, Qyy)
end

"""
    a,b = params2poly(params,na,nb; inputdelay = zeros(Int, size(nb)))

Used to get numerator and denominator polynomials after arx fitting
"""
function params2poly(w, na, nb; inputdelay = zeros(Int, size(nb)))
    maxb = maximum(nb .+ inputdelay)
    a = [1; -w[1:na]]
    a = [a; zeros(max(0, maxb - na))] # if nb > na
    w = w[na+1:end]
    b = map(1:length(nb)) do i
        b = w[1:nb[i]]
        w = w[nb[i]+1:end]
        b = [zeros(inputdelay[i]); b; zeros(maxb - inputdelay[i] - nb[i])] # compensate for different nbs and delay
        b
    end
    return a, b
end
"""
    a,b = params2poly2(params,na,nb; inputdelay = ones(Int, size(nb)))
Used to get numerator and denominator polynomials after arx fitting. Updated version supporting a direct term.
"""
function params2poly2(w, na, nb; inputdelay = ones(Int, size(nb)))
    maxb = maximum(nb .+ inputdelay)
    a = [1; -w[1:na]]
    a = [a; zeros(max(0, maxb - na -1))] # if nb > na
    w = w[na+1:end]
    b = map(1:length(nb)) do i
        b = w[1:nb[i]]
        w = w[nb[i]+1:end]
        b = [zeros(inputdelay[i]); b; zeros(max(na+1, maxb) - inputdelay[i] - nb[i])] # compensate for different nbs and delay, as well as na > nb
        b
    end
    return a, b
end

function params2poly(w, na)
    a = [1; -w]
    b = [1; zeros(na)]
    return a, b
end

# """
# w, a, b = params(G::TransferFunction)
# w = [a;b]
# """
# function params(G::TransferFunction)
#     am, bm = -denvec(G)[1][2:end], numvec(G)[1]
#     wm     = [am; bm]
#     wm, am, bm
# end
"""
w, a, b, inputdelay = params(G::TransferFunction)
w = [a; vcat(b...)]

retrieve na and nb with:
na = length(a)
nb = map(length, vec(b))
"""
function params(G::TransferFunction)
    ams = vec(denvec(G))
    am = -ams[1][2:end]
    filter!(!iszero, am) # assumption, that no coefficient actually iszero

    bm = vec(numvec(G)) 
    inputdelay = length.(ams) .- length.(bm)
    filter!.(!iszero, bm)

    wm = [am; vcat(bm...) ]
    wm, am, bm, inputdelay
end

"""
    Σ = parameter_covariance(y_train, A, w, λ=0)
"""
function parameter_covariance(y_train, A, w, λ = 0)
    σ² = var(y_train .- A * w)
    iATA = if λ == 0
        R = UpperTriangular(qr(A).R)
        R\(R'\I)
    else
        Al = [A; sqrt(λ)*I]
        R = UpperTriangular(qr(Al).R)
        (R \ (R' \ (A'A)) / R) / R' # The parenthesis are important 
    end
    iATA = (iATA + iATA') / 2
    Σ = Hermitian(σ² * iATA + sqrt(eps()) * I)
end
#=
The above formulas using qr factorization can be verified with
A = randn(10,3)
iA = inv(A'A)

QR = qr(A)
iA2 = QR.R\(QR.R'\I)
@test norm(iA2 - iA) / norm(iA) < 1e-13

λ = 0.4
Al = [A; sqrt(λ)*I]
QR = qr(Al)
ATA = A'A
ATAλ = ATA + λ * I
@assert Al'Al ≈ ATAλ
iA = ATAλ \ ATA / ATAλ
iA2 = (QR.R\(QR.R'\(A'A)) / QR.R) / QR.R'

@test norm(iA2 - iA) / norm(iA) < 1e-13
=#

"""
TransferFunction(T::Type{<:AbstractParticles}, G::TransferFunction, Σ, N=500)

Create a `TransferFunction` where the coefficients are `Particles` from [`MonteCarloMeasurements.jl`](https://github.com/baggepinnen/MonteCarloMeasurements.jl) that can represent uncertainty.
# Example
```julia
using MonteCarloMeasurements
G       = ar(d,2,stochastic=true)
w       = exp10.(LinRange(-3,log10(π/Δt),100))
mag     = bode(Glsp,w)[1][:]
errorbarplot(w,mag,0.01; yscale=:log10, xscale=:log10, layout=3, subplot=1, lab="ls")

See full example [here](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/controlsystems.jl)
```
"""
function ControlSystems.TransferFunction(
    T::Type{<:MonteCarloMeasurements.AbstractParticles},
    G::TransferFunction,
    Σ::AbstractMatrix,
    N = 500,
)
    wm, am, bm, inputdelay = params(G)
    na = length(am)
    nb = map(length, vec(bm))
    if length(nb) == 1 ## SISO
        b = bm[1]
        isAR = b[1] == 1 && all(b[2:end] .== 0)
    else
        isAR = false # MISO -> ARX
    end

    if isAR && size(Σ, 1) < length(wm)
        p = T(N, MvNormal(wm[1:end-nb[1]], Matrix(Σ)))
        a, b = params2poly(p, na)
    else
        p = T(N, MvNormal(wm, Matrix(Σ)))
        a, b = params2poly2(p, na, nb, inputdelay = inputdelay)
    end
    arxtf = tf(b, a, G.Ts)
end

# function ControlSystems.TransferFunction(T::Type{<:MonteCarloMeasurements.AbstractParticles}, G::TransferFunction, p::AbstractMatrix)
#       wm, am, bm = params(G)
#       na,nb      = length(am), length(bm)
#       p          = T(p .+ wm')
#       a,b        = params2poly(p,na,nb)
#       arxtf      = tf(b,a,G.Ts)
# end

function DSP.filt(tf::ControlSystems.TransferFunction, y)
    b, a = numvec(tf)[], denvec(tf)[]
    filt(b, a, y)
end
