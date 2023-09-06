module ControlSystemIdentificationLSOptExt

## Nonlinear PEM
using ControlSystemIdentification
using ControlSystemIdentification: AbstractIdData
using LowLevelParticleFilters
using LeastSquaresOptim
using StaticArrays
using LinearAlgebra
using ForwardDiff
using LowLevelParticleFilters.Distributions: MvNormal


"""
    NonlinearPredictionErrorModel{UKF, P, X0}

A nonlinear prediction-error model produced by [`nonlinear_pem`](@ref).

# Fields:
- `ukf::UKF`: The Unscented Kalman Filter used to perform the prediction
- `p::P`: The optimized parameter vector
- `x0::X0`: The optimized initial condition
- `res`: The optimization result structure
- `Λ::Function`: A functor that returns an estimate of the precision matrix (inverse covariance matrix). Several caveats apply to this estimate, use with care.
"""
struct NonlinearPredictionErrorModel{UKF, P, X0}
    ukf::UKF
    p::P
    x0::X0
    res
    Λ::Function
    Ts
end


"""
    nonlinear_pem(d::IdData, discrete_dynamics, measurement, p0, x0, R1, R2, nu; optimizer = LevenbergMarquardt())

Nonlinear Prediction-Error Method (PEM).

This method attempts to find the optimal vector of parameters, ``p``, and the initial condition ``x_0``, that minimizes the sum of squared one-step prediction errors. The prediction is performed using an Unscented Kalman Filter (UKF) and the optimization is performed using a Gauss-Newton method. 

# Arguments:
- `d`: Identification data
- `discrete_dynamics`: A dynamics function `(xₖ, uₖ, p, t) -> x(k+1)` that takes the current state `x`, input `u`, parameters `p`, and time `t` and returns the next state `x(k+1)`.
- `measurement`: The measurement / output function of the nonlinear system `(xₖ, uₖ, p, t) -> yₖ`
- `p0`: The initial guess for the parameter vector
- `x0`: The initial guess for the initial condition
- `R1`: State covariance matrix
- `R2`: Measurement covariance matrix
- `nu`: Number of inputs to the system
- `optimizer`: Any optimizer from [LeastSquaresOptim](https://github.com/matthieugomez/LeastSquaresOptim.jl)
- `λ`: A weighting factor to minimize `dot(e, λ, e`). A commonly used metric is `λ = Diagonal(1 ./ (mag.^2))`, where `mag` is a vector of the "typical magnitude" of each output. Internally, the square root of `W = sqrt(λ)` is calculated so that the residuals stored in `res` are `W*e`.

The inner optimizer accepts a number of keyword arguments:
- `lower`: Lower bounds for the parameters
- `upper`: Upper bounds for the parameters
- `x_tol = 1e-8`
- `f_tol = 1e-8`
- `g_tol = 1e-8`
- `iterations = 1_000`
- `Δ = 10.0`
- `store_trace = false`

!!! warning "Experimental"
    This function is considered experimental and may change in the future without respecting semantic versioning. This implementation also lacks a number of features associated with good nonlinear PEM implementations, such as regularization and support for multiple datasets.
"""
function ControlSystemIdentification.nonlinear_pem(d::AbstractIdData, discrete_dynamics, measurement, p0::AbstractVector, x0::AbstractVector, R1::AbstractMatrix, R2::AbstractMatrix, nu::Int; optimizer = LevenbergMarquardt(), λ=1, kwargs...)

    nx = size(R1, 1)
    ny = size(R2, 1)

    y = output(d)
    u = input(d)

    ET = eltype(y)

    yvv = reinterpret(SVector{ny, ET}, y)
    uvv = reinterpret(SVector{nu, ET}, u)

    R1 = SMatrix{nx, nx, ET, nx^2}(R1)
    R2 = SMatrix{ny, ny, ET, ny^2}(R2)

    x0inds = SVector{nx, Int}((1:nx) .+ length(p0))

    _inner_pem(yvv, uvv, x0inds, discrete_dynamics, measurement, p0, x0, nu, R1, R2, optimizer, λ; kwargs...) 
end

# Function barrier to handle the type instability caused by the static arrays above
function _inner_pem(yvv, uvv, x0inds, discrete_dynamics::F, measurement::G, p0, x0, nu, R1, R2, optimizer, λ; kwargs...) where {F,G}
    R1mut = Matrix(R1)

    function get_ukf(px0::Vector{T}) where T
        pᵢ = px0[1:length(p0)]
        x0i = px0[x0inds]
        UnscentedKalmanFilter(discrete_dynamics, measurement, R1, R2, MvNormal(T.(x0i), R1mut); ny, nu, p=pᵢ)
    end

    function residuals!(ϵ, px0)
        ukf = get_ukf(px0)
        pᵢ = px0[1:length(p0)]
        LowLevelParticleFilters.prediction_errors!(ϵ, ukf, uvv, yvv, pᵢ, λ)
    end

    p_guess = [p0; x0]
    T = length(yvv)
    ny = size(R2, 1)

    res = optimize!(LeastSquaresProblem(; x = p_guess, f! = residuals!, output_length = T*ny, autodiff = :forward), optimizer; show_trace=true, show_every=1, kwargs...) 

    p = res.minimizer[1:length(p0)]
    x0 = res.minimizer[x0inds]

    function Λ()
        resid = zeros(T*ny)
        J = ForwardDiff.jacobian(residuals!, resid, res.minimizer)
        (T-length(p_guess)) * Symmetric(J'*J)
    end

    ukf = get_ukf(res.minimizer)

    NonlinearPredictionErrorModel(ukf, p, x0, res, Λ, d.Ts)
end



function LowLevelParticleFilters.simulate(model::NonlinearPredictionErrorModel, u, x0 = model.x0; p = model.ukf.p)
    reset!(model.ukf; x0)
    @show p
    if u isa AbstractIdData
        u = input(u)
    end
    _,_,y = LowLevelParticleFilters.simulate(model.ukf, collect.(eachcol(u)), p; dynamics_noise=false, measurement_noise=false)
    reduce(hcat, y)
end

function ControlSystemIdentification.predict(model::NonlinearPredictionErrorModel, d::AbstractIdData, x0 = model.x0; p = model.p, h=1)
    # model.Ts == d.Ts || throw(ArgumentError("Sample time mismatch between data $(d.Ts) and system $(model.Ts)"))
    h == 1 || throw(ArgumentError("Only h=1 is supported at the moment"))
    reset!(model.ukf; x0)
    sol = forward_trajectory(model.ukf, collect.(eachcol(d.u)), collect.(eachcol(d.y)), p)
    reduce(hcat, sol.y)
end

ControlSystemIdentification.get_x0(x0::Nothing, sys::NonlinearPredictionErrorModel, d::AbstractIdData) = sys.x0
ControlSystemIdentification.get_x0(x0::Symbol, sys::NonlinearPredictionErrorModel, d::AbstractIdData) = sys.x0


end