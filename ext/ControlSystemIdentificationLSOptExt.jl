module ControlSystemIdentificationLSOptExt

## Nonlinear PEM
using ControlSystemIdentification
using ControlSystemIdentification: AbstractIdData
import ControlSystemIdentification: nonlinear_pem
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
struct NonlinearPredictionErrorModel{UKF,P,X0}
    ukf::UKF
    p::P
    x0::X0
    res::Any
    Λ::Function
    Ts::Any
end

function Base.show(io::IO, model::NonlinearPredictionErrorModel)
    println(io, "NonlinearPredictionErrorModel")
    println(io, "  p: ", model.p)
    println(io, "  x0: ", model.x0)
    println(io, "  Ts: ", model.Ts)
    println(io, "  ny = $(model.ukf.ny), nu = $(model.ukf.nu), nx = $(model.ukf.nx)")
end



function nonlinear_pem(
    d::AbstractIdData,
    discrete_dynamics,
    measurement,
    p0::AbstractVector,
    x0::AbstractVector,
    R1::AbstractMatrix,
    R2::AbstractMatrix,
    nu::Int;
    optimizer = LevenbergMarquardt(),
    λ = 1.0,
    optimize_x0 = true,
    kwargs...,
)

    nx = size(R1, 1)
    ny = size(R2, 1)

    y = output(d)
    u = input(d)

    ET = eltype(y)

    yvv = reinterpret(SVector{ny,ET}, y)
    uvv = reinterpret(SVector{nu,ET}, u)

    R1 = SMatrix{nx,nx,ET,nx^2}(R1)
    R2 = SMatrix{ny,ny,ET,ny^2}(R2)

    if optimize_x0
        x0inds = SVector{nx,Int}((1:nx) .+ length(p0))
    else
        x0inds = SVector{0,Int}()
    end
    x0 = SVector(x0...)

    _inner_pem(d.Ts, yvv, uvv, x0inds, discrete_dynamics, measurement, p0, x0, nu,
        R1, R2, optimizer, λ, optimize_x0; kwargs...) 
end

"""
    nonlinear_pem(d, model::NonlinearPredictionErrorModel; x0, R1, R2, kwargs...)

Nonlinear Prediction-Error Method initialized with a model resulting from a previous call to `nonlinear_pem`. Parameters `x0, R1, R2` can be modified to refine the optimization. Calling `nonlinear_pem` repeatedly may be beneficial when the first call uses a small `R2` for large amounts of measurement feedback, this makes it easier to find a good model when the initial parameter guess is poor. A second call can use a larger `R2` to improve the simulation performance of the estimated model.
"""
function nonlinear_pem(d::AbstractIdData, model::NonlinearPredictionErrorModel; x0 = model.x0, R1=model.ukf.R1, R2=model.ukf.R2, kwargs...)
    nonlinear_pem(d, model.ukf.dynamics, model.ukf.measurement, model.p, x0, R1, R2, model.ukf.nu; kwargs...)
end

# Function barrier to handle the type instability caused by the static arrays above
function _inner_pem(
    Ts,
    yvv,
    uvv,
    x0inds,
    discrete_dynamics::F,
    measurement::G,
    p0,
    x0,
    nu,
    R1,
    R2,
    optimizer,
    λ,
    optimize_x0;
    autodiff = :forward,
    kwargs...,
) where {F,G}
    R1mut = Matrix(R1)

    function get_ukf(px0::Vector{T}) where {T}
        pᵢ = px0[1:length(p0)]
        x0i = optimize_x0 ? px0[x0inds] : x0
        UnscentedKalmanFilter(discrete_dynamics, measurement, R1, R2, MvNormal(T.(x0i), R1mut); ny, nu, p=pᵢ)
    end

    function residuals!(ϵ, px0)
        ukf = get_ukf(px0)
        pᵢ = px0[1:length(p0)]
        LowLevelParticleFilters.prediction_errors!(ϵ, ukf, uvv, yvv, pᵢ, λ)
    end

    if optimize_x0
        p_guess = [p0; x0]
    else
        p_guess = copy(p0)
    end
    T = length(yvv)
    ny = size(R2, 1)

    res = optimize!(
        LeastSquaresProblem(;
            x = p_guess,
            f! = residuals!,
            output_length = T * ny,
            autodiff,
        ),
        optimizer;
        show_trace = true,
        show_every = 1,
        kwargs...,
    )

    p = res.minimizer[1:length(p0)]
    x0 = optimize_x0 ? res.minimizer[x0inds] : x0

    function Λ()
        resid = zeros(T * ny)
        J = ForwardDiff.jacobian(residuals!, resid, res.minimizer)
        (T - length(p_guess)) * Symmetric(J' * J)
    end

    ukf = get_ukf(res.minimizer)

    NonlinearPredictionErrorModel(ukf, p, x0, res, Λ, Ts)
end


function LowLevelParticleFilters.simulate(model::NonlinearPredictionErrorModel, u, x0 = model.x0; p = model.ukf.p)
    reset!(model.ukf; x0)
    if u isa AbstractIdData
        u = input(u)
    end
    _,_,y = LowLevelParticleFilters.simulate(model.ukf, collect.(eachcol(u)), p; dynamics_noise=false, measurement_noise=false)
    reduce(hcat, y)
end

function ControlSystemIdentification.predict(model::NonlinearPredictionErrorModel, d::AbstractIdData, x0 = model.x0; p = model.p, h=1)
    # model.Ts == d.Ts || throw(ArgumentError("Sample time mismatch between data $(d.Ts) and system $(model.Ts)"))
    h == 1 || h == 0 || throw(ArgumentError("Only h=1 is supported at the moment"))
    reset!(model.ukf; x0)
    sol = forward_trajectory(model.ukf, d, p)
    yh = model.ukf.measurement.(h == 1 ? sol.x : sol.xt, eachcol(d.u), Ref(p), 1:length(d))
    reduce(hcat, yh)
end

ControlSystemIdentification.get_x0(x0::Nothing, sys::NonlinearPredictionErrorModel, d::AbstractIdData) = sys.x0
ControlSystemIdentification.get_x0(x0::Symbol, sys::NonlinearPredictionErrorModel, d::AbstractIdData) = sys.x0


end
