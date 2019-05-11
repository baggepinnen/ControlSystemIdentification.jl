"""
    getARXregressor(y::AbstractVector,u::AbstractVecOrMat, na, nb)
Returns a shortened output signal `y` and a regressor matrix `A` such that the least-squares ARX model estimate of order `na,nb` is `y\\A`
Return a regressor matrix used to fit an ARX model on, e.g., the form
`A(z)y = B(z)f(u)`
with output `y` and input `u` where the order of autoregression is `na` and
the order of input moving average is `nb`
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
function getARXregressor(y::AbstractVector,u::AbstractVecOrMat, na, nb)
    length(nb) == size(u,2) || throw(ArgumentError("Length of nb must equal number of input signals"))
    m    = max(na,maximum(nb))+1 # Start of yr
    @assert m >= 1
    n    = length(y) - m + 1 # Final length of yr
    @assert n <= length(y)
    A    = toeplitz(y[m:m+n-1],y[m:-1:m-na])
    @assert size(A,2) == na+1
    y    = A[:,1] # extract yr
    A    = A[:,2:end]
    for i = 1:length(nb)
        nb[i] <= 0 && continue
        s = m-1
        A = [A toeplitz(u[s:s+n-1,i],u[s:-1:s-nb[i]+1,i])]
    end
    return y,A
end

function getARregressor(y::AbstractVector, na)
    m    = na+1 # Start of yr
    n    = length(y) - m + 1 # Final length of yr
    A    = toeplitz(y[m:m+n-1],y[m:-1:m-na])
    @assert size(A,2) == na+1
    y    = A[:,1] # extract yr
    A    = A[:,2:end]
    return y,A
end

@userplot Find_na
"""
    find_na(y::AbstractVector,n::Int)
Plots the RMSE and AIC For model orders up to `n`. Useful for model selection
"""
find_na
@recipe function find_na(p::Find_na)
    y,n = p.args[1:2]
    error = zeros(n,2)
    for i = 1:n
        yt,A = getARXregressor(y,0y,i,0)
        e = yt-A*(A\yt)
        error[i,1] = rms(e)
        error[i,2] = aic(e,i)
    end
    layout --> 2
    title --> ["RMS error" "AIC"]
    seriestype --> :scatter
    @series begin
        error
    end
end

@userplot Find_nanb
"""
    find_nanb(y::AbstractVector,u,na,nb)
Plots the RMSE and AIC For model orders up to `n`. Useful for model selection
"""
find_nanb
@recipe function find_nanb(p::Find_nanb; logrms=false)
    y,u,na,nb = p.args[1:4]
    error = zeros(na,nb, 2)
    for i = 1:na, j = 1:nb
        yt,A = getARXregressor(y,u,i,j)
        e = yt-A*(A\yt)
        error[i,j,1] = logrms ? log10.(rms(e)) : rms(e)
        error[i,j,2] = aic(e,i+j)
    end
    layout --> 2
    seriestype --> :heatmap
    xticks := (1:nb, 1:nb)
    yticks := (1:na, 1:na)
    ylabel := "na"
    xlabel := "nb"
    @series begin
        title := "RMS error"
        subplot := 1
        error[:,:,1]
    end
    @series begin
        title := "AIC"
        subplot := 2
        error[:,:,2]
    end
end



"""
    Gtf, Σ = arx(h,y, u, na, nb; λ = 0, estimator=\\)

Fit a transfer Function to data using an ARX model and equation error minimization.
`nb` and `na` are the length of the numerator and denominator polynomials. `h` is the sample time of the data. `λ > 0` can be provided for L₂ regularization. `estimator` defaults to \\ (least squares), alternatives are `estimator = tls` for total least-squares estimation. `arx(Δt,yn,u,na,nb, estimator=wtls_estimator(y,na,nb)` is potentially more robust in the presence of heavy measurement noise.
The number of free parameters is `na-1+nb`
`Σ` is the covariance matrix of the parameter estimate. See `bodeconfidence` for visualiztion of uncertainty.

Supports MISO estimation by supplying a matrix `u` where times is first dim, with nb = [nb₁, nb₂...]
"""
function arx(h,y::AbstractVector, u::AbstractVecOrMat, na, nb; λ = 0, estimator=\)
    all(nb .<= na) || throw(DomainError(nb,"nb must be <= na"))
    na >= 1 || throw(ArgumentError("na must be positive"))
    y_train, A = getARXregressor(y,u, na, nb)
    w = ls(A,y_train,λ,estimator)
    a,b = params2poly(w,na,nb)
    model = tf(b,a,h)
    Σ = parameter_covariance(y_train, A, w, λ)
    return model, Σ
end

function ar(h,y::AbstractVector, na; λ = 0, estimator=\)
    na >= 1 || throw(ArgumentError("na must be positive"))
    y_train, A = getARregressor(y, na)
    w = ls(A,y_train,λ,estimator)
    a,b = params2poly(w,na)
    model = tf(b,a,h)
    Σ = parameter_covariance(y_train, A, w, λ)
    return model, Σ
end

function ls(A,y,λ=0,estimator=\)
    if λ == 0
        w = estimator(A,y)
    else
        w = estimator([A; λ*I], [y;zeros(size(A,2))])
    end
    w
end

"""
G, Gn = plr(h,y,u,na,nb,nc; initial_order = 20)

Perform pseudo-linear regression to estimate a model on the form
`Ay = Bu + Cw`
The residual sequence is estimated by first estimating a high-order arx model, whereafter the estimated residual sequence is included in a second estimation problem. The return values are the estimated system model, and the estimated noise model. `G` and `Gn` will always have the same denominator polynomial.
"""
function plr(h,y,u,na,nb,nc; initial_order = 20)
    all(nb .<= na) || throw(DomainError(nb,"nb must be <= na"))
    na >= 1 || throw(ArgumentError("na must be positive"))
    na -= 1
    y_train, A = getARXregressor(y,u,initial_order,initial_order)
    w1 = A\y_train
    yhat = A*w1
    ehat = yhat - y_train
    ΔN = length(y)-length(ehat)
    size(u), size(ehat)
    y_train, A = getARXregressor(y[ΔN+1:end-1],[u[ΔN+1:end-1,:] ehat[1:end-1]],na,[nb;nc])
    w = A\y_train
    a,b = params2poly(w,na,nb)
    model = tf(b,a,h)
    c = w[na+sum(nb)+1:end]
    noise_model = tf(c,a,h)
    model, noise_model
end
# Helper constructor to make a MISO system after MISO arx estimation
function ControlSystems.tf(b::AbstractVector{<:AbstractVector{<:Number}}, a::AbstractVector{<:Number}, h)
    tfs = map(b) do b
        tf(b,a,h)
    end
    hcat(tfs...)
end

"""
wtls_estimator(y,na,nb)
Create an estimator function for estimation of arx models in the presence of measurement noise.
"""
function wtls_estimator(y,na,nb)
    rowQ = [diagm(0=>[ones(na);zeros(nb);1]) for i = 1:length(y)-(na)]
    Qaa,Qay,Qyy = rowcovariance(rowQ)
    (A,y)->wtls(A,y,Qaa,Qay,Qyy)
end

"""
    a,b = params2poly(params,na,nb)
Used to get numerator and denominator polynomials after arx fitting
"""
function params2poly(w,na,nb)
    a = [1; -w[1:na]]
    w = w[na+1:end]
    b = map(nb) do nb
        b = w[1:nb]
        w = w[nb+1:end]
        b
    end
    a,b
end
function params2poly(w,na)
    a = [1; -w]
    b = 1
    a,b
end
"""
w, a, b = params(G::TransferFunction)
w = [a;b]
"""
function params(G::TransferFunction)
    am, bm = -denpoly(G)[1].a[1:end-1], G.matrix[1].num.a
    wm     = [am; bm]
    wm, am, bm
end

"""
    Σ = parameter_covariance(y_train, A, w, λ=0)
"""
function parameter_covariance(y_train, A, w, λ=0)
    σ² = var(y_train .- A*w)
    iATA = if λ == 0
        inv(A'A)
    else
        ATA = A'A
        ATAλ = ATA + λ*I
        ATAλ\ATA/ATAλ
    end
    iATA = (iATA+iATA')/2
    Σ = σ²*iATA + sqrt(eps())*Matrix(LinearAlgebra.I,size(iATA))
end

"""
TransferFunction(T::Type{<:AbstractParticles}, G::TransferFunction, Σ, N=500)

Create a `TransferFunction` where the coefficients are `Particles` from [`MonteCarloMeasurements.jl`](https://github.com/baggepinnen/MonteCarloMeasurements.jl) that can represent uncertainty.
# Example
```julia
using MonteCarloMeasurements
Gls,Σls = arx(Δt,y,u,na,nb)
Glsp    = TransferFunction(Particles, Gls, Σls)
w       = exp10.(LinRange(-3,log10(π/Δt),100))
mag     = bode(Glsp,w)[1][:]
errorbarplot(w,mag,0.01; yscale=:log10, xscale=:log10, layout=3, subplot=1, lab="ls")

See full example [here](https://github.com/baggepinnen/MonteCarloMeasurements.jl/blob/master/examples/controlsystems.jl)
```
"""
function ControlSystems.TransferFunction(T::Type{<:MonteCarloMeasurements.AbstractParticles}, G::TransferFunction, Σ, N=500)
      wm, am, bm = ControlSystemIdentification.params(G)
      na,nb  = length(am), length(bm)
      p = T(N, MvNormal(wm, Σ))
      a,b           = ControlSystemIdentification.params2poly(p,na,nb)
      arxtf         = tf(b,a,G.Ts)
end

function DSP.filt(tf::ControlSystems.TransferFunction, y)
    b,a = numvec(tf)[], denvec(tf)[]
    filt(b,a,y)
end


"""
    bodeconfidence(arxtf::TransferFunction, Σ::Matrix, ω = logspace(0,3,200))
Plot a bode diagram of a transfer function estimated with [`arx`](@ref) with confidence bounds on magnitude and phase.
"""
bodeconfidence

@userplot BodeConfidence

@recipe function BodeConfidence(p::BodeConfidence)
    arxtfm = p.args[1]
    Σ      = p.args[2]
    ω      = length(p.args) >= 3 ? p.args[3] : exp10.(LinRange(-2,3,200))
    L      = cholesky(Hermitian(Σ)).L
    wm, am, bm = params(arxtfm)
    na,nb  = length(am), length(bm)
    mc     = 100
    res = map(1:mc) do _
        w             = L*randn(size(L,1)) .+ wm
        a,b           = params2poly(w,na,nb)
        arxtf         = tf(b,a,arxtfm.Ts)
        mag, phase, _ = bode(arxtf, ω)
        mag[:], phase[:]
    end
    magmc      = reduce(hcat, getindex.(res,1))
    phasemc    = reduce(hcat, getindex.(res,2))
    mag        = mean(magmc,dims=2)[:]
    phase      = mean(phasemc,dims=2)[:]
    # mag,phase,_ = bode(arxtfm, ω) .|> x->x[:]
    uppermag   = getpercentile(magmc,0.95)[:]
    lowermag   = getpercentile(magmc,0.05)[:]
    upperphase = getpercentile(phasemc,0.95)[:]
    lowerphase = getpercentile(phasemc,0.05)[:]
    layout := (2,1)

    @series begin
        subplot := 1
        title --> "ARX estimate"
        ylabel --> "Magnitude"
        fillrange := (lowermag, uppermag)
        yscale --> :log10
        xscale --> :log10
        alpha --> 0.3
        ω, mag
    end
    @series begin
        subplot := 2
        fillrange := (lowerphase, upperphase)
        ylabel --> "Phase [deg]"
        xlabel --> "Frequency [rad/s]"
        xscale --> :log10
        alpha --> 0.3
        ω, phase
    end

end

"""
    getpercentile(x,p)

calculates the `p`th percentile along dim 2
"""
function getpercentile(mag,p)
    uppermag = mapslices(mag, dims=2) do magω
        sort(magω)[round(Int,length(magω)*p)]
    end
end
