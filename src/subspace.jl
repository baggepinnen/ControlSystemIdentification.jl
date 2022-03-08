@static if VERSION < v"1.3"
    (LinearAlgebra.I)(n) = Matrix{Float64}(I, n, n)
end

"""
    res = n4sid(data, r=:auto; verbose=false)

Estimate a statespace model using the n4sid method. Returns an object of type [`N4SIDStateSpace`](@ref) where the model is accessed as `res.sys`.

Implements the simplified algorithm (alg 2) from
"N4SID: Subspace Algorithms for the Identification of Combined Deterministic Stochastic Systems" PETER VAN OVERSCHEE and BART DE MOOR

The frequency weighting is borrowing ideas from
"Frequency Weighted Subspace Based System Identication in the Frequency Domain", Tomas McKelvey 1996. In particular, we apply the output frequency weight matrix (Fy) as it appears in eqs. (16)-(18).

# Arguments:
- `data`: Identification data `data = iddata(y,u)`
- `r`: Rank of the model (model order)
- `verbose`: Print stuff?
- `Wf`: A frequency-domain model of measurement disturbances. To focus the attention of the model on a narrow frequency band, try something like `Wf = Bandstop(lower, upper, fs=1/Ts)` to indicate that there are disturbances *outside* this band.
- `i`: Algorithm parameter, generally no need to tune this
- `γ`: Set this to a value between (0,1) to stabilize unstable models such that the largest eigenvalue has magnitude γ.
- `zeroD`: defaults to false
"""
function n4sid(
    data::InputOutputData,
    r = :auto;
    verbose = false,
    i = r === :auto ? min(length(data) ÷ 20, 20) : r + 10,
    γ = nothing,
    Wf = nothing,
    zeroD = false,
    svd::F1 = svd,
    svd!::F2 = svd!,
    estimator::F3 = \,
) where {F1,F2,F3}

    y, u = time1(output(data)), time1(input(data))
    N, l = size(y, 1), size(y, 2)
    m = size(u, 2)
    j = N - 2i

    function hankel(u::AbstractArray, i1, i2)
        d = size(u, 2)
        w = (i2 - i1 + 1)
        H = zeros(eltype(u), w * d, j)
        for r = 1:w, c = 1:j
            H[(r-1)*d+1:r*d, c] = u[i1+r+c-1, :]
        end
        H
    end
    mi = m * i
    li = l * i
    U0im1 = hankel(u, 0, i - 1)
    Y0im1 = hankel(y, 0, i - 1)
    Y0i = hankel(y, 0, i)
    U0i = hankel(u, 0, i)
    UY0 = [U0im1; hankel(u, i, 2i - 1); Y0im1]
    UY1 = [U0i; hankel(u, i + 1, 2i - 1); Y0i]
    # proj(A, B) = A * (B' / (B * B'))
    # proj(A, B) = (A * B') / (B * B')
    proj(A, B) = (svd!(B * B') \ (A * B')')' # todo: these can be improved by not forming BB', use QR
    Li = proj(hankel(y, i, 2i - 1), UY0)
    Lip1 = proj(hankel(y, i + 1, 2i - 1), UY1)

    L¹ᵢ = Li[:, 1:mi]
    L³ᵢ = Li[:, end-li+1:end]

    L¹ᵢp1 = Lip1[:, 1:m*(i+1)]
    L³ᵢp1 = Lip1[:, end-l*(i+1)+1:end]

    # Zi = Li*UY0
    # Zip1 = Lip1*UY1

    Zi = [L¹ᵢ L³ᵢ] * [U0im1; Y0im1]

    if Wf === nothing
        s = svd(Zi)
    else
        W = frequency_weight(Wf, size(Zi, 1))
        s = svd!(W \ Zi)
        estimator = weighted_estimator(Wf)
    end
    if r === :auto
        r = sum(s.S .> sqrt(s.S[1] * s.S[end]))
        verbose && @info "Choosing order $r"
    end
    n = r
    U1 = s.U[:, 1:r]
    S1 = s.S[1:r]
    fve = sum(S1) / sum(s.S)
    verbose && @info "Fraction of variance explained: $(fve)"

    Γi = U1 * Diagonal(sqrt.(S1))
    if Wf !== nothing
        Γi = W * Γi
    end
    Γim1 = Γi[1:end-l, :]
    Xi = estimator(Γi, Zi)
    Xip1 = estimator(Γim1, [L¹ᵢp1 L³ᵢp1] * [U0i; Y0i])

    XY = [Xip1; hankel(y, i, i)]
    XU = [Xi; hankel(u, i, i)]
    if zeroD
        L = estimator(XU', Xip1')'
        C = estimator(Xi', hankel(y, i, i)')'
        D = zeros(eltype(C), l, m)
        L = [L; [C D]]
    else
        L = estimator(XU', XY')'
        C = L[n+1:end, 1:n]
        D = L[n+1:end, n+1:end]
    end

    A = L[1:n, 1:n]
    if γ !== nothing
        e = maximum(abs, eigvals(A))
        if e > γ
            verbose && @info "Stabilizing A, maximum eigval had magnitude $e"
            L = stabilize(L, XU, i, j, m, n, γ)
            A = L[1:n, 1:n]
        end
    end
    B = L[1:n, n+1:end]


    errors = XY - L * XU
    Σ = 1 / (j - (n + 1)) * errors * errors'
    Q = Symmetric(Σ[1:n, 1:n])
    R = Symmetric(Σ[n+1:end, n+1:end])
    S = Σ[1:n, n+1:end]

    local P, K
    try
        P, _, Kt, _ = MatrixEquations.ared(copy(A'), copy(C'), R, Q, S)
        K = Kt' |> copy
    catch
        P = fill(NaN, n, n)
        K = fill(NaN, n, l)
    end

    sys = ss(A, B, C, D, sampletime(data))

    N4SIDStateSpace(sys, Q, R, S, K, P, Xi, s.S, fve)
end



"""
"Imposing Stability in Subspace Identification by Regularization", Gestel, Suykens, Dooren, Moor
"""
function stabilize(L, XU, i, j, m, n, γ)
    UtXt = XU[[n+1:end; 1:n], :]'

    F   = qr(UtXt)
    R22 = F.R[m+1:end, m+1:end]
    Σ   = R22'R22
    P2  = -γ^2 * I(n^2)
    P1  = -γ^2 * kron(I(n), Σ) - γ^2 * kron(Σ, I(n))
    A   = L[1:n, 1:n]
    AΣ  = A * Σ
    P0  = kron(AΣ, AΣ) - γ^2 * kron(Σ, Σ)

    θ = eigvals(Matrix([0I(n^2) -I(n^2); P0 P1]), -Matrix([I(n^2) 0I(n^2); 0I(n^2) P2]))
    c = maximum(abs.(θ[(imag.(θ).==0).*(real.(θ).>0)]))

    Σ_XU = Symmetric(XU * XU')
    mod = Σ_XU / (Σ_XU + c * Diagonal([ones(n); zeros(m)]))#[I(n) zeros(n,m); zeros(m, n+m)])
    L[1:n, :] .= L[1:n, :] * mod
    L
end


function sysfilter!(state::AbstractVector, res::AbstractPredictionStateSpace{<:Discrete}, y, u)
    @unpack A, B, C, D, ny, sys, K = res
    yh = vec(C * state + D * u)
    e = y - yh
    state .= vec(A * state + B * u + K * e)
    yh
end

function sysfilter!(state::AbstractVector, res::AbstractPredictionStateSpace{<:Discrete}, u)
    @unpack A, B, C, D, ny, K, sys = res
    yh = vec(C * state + D * u)
    state .= vec(A * state + B * u)
    yh
end

SysFilter(res::AbstractPredictionStateSpace{<:Discrete}, x0 = res.x[:, 1]) = SysFilter(res, x0, res.C * x0)

function simulate(
    res::N4SIDStateSpace,
    d::AbstractIdData,
    x0 = res.x[:, 1];
    stochastic = false,
)
    sys = res.sys
    @unpack A, B, C, D, ny, K, Q, R, P, sys = res
    kf = KalmanFilter(res, x0)
    u = input(d)
    yh = map(observations(u, u)) do (ut, _)
        yh = vec(C * state(kf) + D * ut)
        LowLevelParticleFilters.predict!(kf, ut)
        stochastic ?
        StaticParticles(MvNormal(yh, Symmetric(C * covariance(kf) * C' + kf.R2))) : yh
    end
    oftype(u, yh)
end

function predict(sys::AbstractPredictionStateSpace, d::AbstractIdData, x0 = nothing; h=1)
    sys.Ts == d.Ts || throw(ArgumentError("Sample time mismatch between data $(d.Ts) and system $(sys.Ts)"))
    pd = predictiondata(d)
    pred = observer_predictor(sys; h)
    
    if x0 === nothing
        x0 = estimate_x0(pred, pd)
    else
        x0 = estimate_x0(pred, pd, fixed=[fill(Inf, pred.nx-length(x0)); x0])
    end
    
    lsim(pred, pd.u; x0).y
end

Base.promote_rule(::Type{StateSpace{TE}}, ::Type{<:AbstractPredictionStateSpace{TE}}) where TE<:Discrete = StateSpace{TE}
Base.promote_rule(::Type{<:StateSpace{TE}}, ::Type{N4SIDStateSpace}) where TE<:Discrete = StateSpace{TE}

function Base.convert(::Type{StateSpace{TE}}, s::AbstractPredictionStateSpace{TE}) where TE<:Discrete
    deepcopy(s.sys)
end

function LowLevelParticleFilters.KalmanFilter(res::AbstractPredictionStateSpace, x0 = zeros(res.nx))
    sys = res.sys
    @unpack A, B, C, D, ny, K, Q, R = res
    if hasproperty(res, :P)
        P = res.P
    else
        P = Matrix(Q)
    end
    iszero(res.S) || @warn "Cross-covariance sys.S is ignored when forming the Kalman filter. The infinite horizon Kalman gain will be $(kalman(sys, Q, R)) instead of $(kalman(sys, Q, R, res.S))"
    kf = KalmanFilter(A, B, C, D, Matrix(Q), Matrix(R), MvNormal(x0, P)) # NOTE: cross covariance S ignored
end


function LowLevelParticleFilters.forward_trajectory(kf::LowLevelParticleFilters.AbstractFilter, d::AbstractIdData)
    y = time2(output(d))
    u = input(d)
    U = m2vv(u)
    Y = m2vv(y)
    forward_trajectory(kf, U, Y)
end

function LowLevelParticleFilters.smooth(kf::LowLevelParticleFilters.AbstractFilter, d::AbstractIdData)
    y = time2(output(d))
    u = input(d)
    U = m2vv(u)
    Y = m2vv(y)
    LowLevelParticleFilters.smooth(kf, U, Y)
end

function LowLevelParticleFilters.smooth(kf::LowLevelParticleFilters.AbstractFilter, M::Int, d::AbstractIdData)
    y = time2(output(d))
    u = input(d)
    U = m2vv(u)
    Y = m2vv(y)
    LowLevelParticleFilters.smooth(kf, M, U, Y)
end

function ControlSystems.balreal(sys::AbstractPredictionStateSpace, args...; kwargs...)
    sysr, G, T = balreal(sys.sys, args...; kwargs...)
    nx = sysr.nx
    K = T*sys.K # similarity transform is reversed here compared to below
    Q = Hermitian(T*something(sys.Q, zeros(nx, nx))*T') # should be transpose here, not inverse
    PredictionStateSpace(sysr, K, Q, sys.R), G, T
end

function ControlSystems.baltrunc(sys::AbstractPredictionStateSpace, args...; kwargs...)
    sysr, G, T = baltrunc(sys.sys, args...; kwargs...)
    nx = sysr.nx
    K = (T*sys.K)[1:nx, :] # similarity transform is reversed here compared to below
    Q = (T*something(sys.Q, zeros(nx, nx))*T')[1:nx, 1:nx]
    PredictionStateSpace(sysr, K, Q, sys.R), G, T
end

function ControlSystems.similarity_transform(sys::AbstractPredictionStateSpace, T; kwargs...)
    syss = similarity_transform(sys.sys, T; kwargs...)
    nx = syss.nx
    K = T\sys.K
    T = pinv(T)
    Q = Hermitian(T*something(sys.Q, zeros(nx, nx))*T') # should be transpose here, not inverse
    PredictionStateSpace(syss, K, Q, sys.R)
end

##
"""
    era(YY::AbstractArray{<:Any, 3}, Ts, r::Int, m::Int, n::Int)

Eigenvalue realization algorithm.

# Arguments:
- `YY`: Markov parameters (impulse response) size `n_out×n_in×n_time`
- `Ts`: Sample time
- `r`: Model order
- `m`: Number of rows in Hankel matrix
- `n`: Number of columns in Hankel matrix
"""
function era(YY::AbstractArray{<:Any,3}, Ts, r::Int, m::Int, n::Int)
    nout, nin, T = size(YY)
    size(YY, 3) >= m + n ||
        throw(ArgumentError("hankel size too large for input size. $(size(YY,3)) < m+n ($(m+n))"))

    Dr = similar(YY, nout, nin)
    Y = similar(YY, nout, nin, T - 1)
    for i = 1:nout
        for j = 1:nin
            Dr[i, j] = YY[i, j, 1]
            Y[i, j, :] = YY[i, j, 2:end]
        end
    end
    H, H2 = zeros(eltype(YY), m * nout, n * nin), zeros(eltype(YY), m * nout, n * nin)
    for i = 1:m
        for j = 1:n
            for Q = 1:nout
                for P = 1:nin
                    i1 = nout * (i - 1) + Q
                    i2 = nin * (j - 1) + P
                    H[i1, i2] = Y[Q, P, i+j-1]
                    H2[i1, i2] = Y[Q, P, i+j]
                end
            end
        end
    end
    # return H
    any(!isfinite, H) && error("Got infinite stuff in H")
    U, S, V = svd(H)
    Ur = U[:, 1:r]
    Vr = V[:, 1:r]
    S2 = Diagonal(1 ./ sqrt.(S[1:r]))
    Ar = S2 * Ur'H2 * Vr * S2
    Br = S2 * Ur'H[:, 1:nin]
    Cr = H[1:nout, :] * Vr * S2
    ss(Ar, Br, Cr, Dr, Ts === nothing ? 1 : Ts)
end

"""
    era(d::AbstractIdData, r, m = 2r, n = 2r, l = 5r; p = l, λ=0)

Eigenvalue realization algorithm. Uses `okid` to find the Markov parameters as an initial step.

# Arguments:
- `r`: Model order
- `l`: Number of Markov parameters to estimate.
- `λ`: Regularization parameter
- `p`: Optionally, delete the first `p` columns in the internal Hankel matrices to account for initial conditions != 0. If `x0 != 0`, for `era`, `p` defaults to `l`, while when calling `okid` directly, `p` defaults to 0.
"""
era(d::AbstractIdData, r, m = 2r, n = 2r, l = 5r; p = l, kwargs...) =
    era(okid(d, r, l; p, kwargs...), d.Ts, r, m, n)


"""
    H = okid(d::AbstractIdData, nx, l = 5nx; p = 1, λ=0, estimator = /)

Observer Kalman filter identification. Returns the Markov parameters `H` size `n_out×n_in×l+1`

# Arguments:
- `nx`: Model order
- `l`: Number of Markov parameters to estimate.
- `λ`: Regularization parameter
- `p`: Optionally, delete the first `p` columns in the internal Hankel matrices to account for initial conditions != 0. If `x0 != 0`, try setting `p` around the same value as `l`.
"""
@views function okid(d::AbstractIdData, nx, l = 5nx; λ = 0, p = 1, estimator = /)
    y, u = time2(output(d)), time2(input(d))
    q, N = size(y) # p is the number of outputs
    m = size(u, 1) # q is the number of inputs
    # Step 2, form y, V, solve for observer Markov params, Ȳ
    V = zeros(eltype(y), m + (m + q) * l, N)
    for i = 1:N
        V[1:m, i] = u[1:m, i]
    end
    for i = 2:l+1
        for j = 1:N+1-i
            vtemp = [u[:, j]; y[:, j]]
            V[m+(i-2)*(m+q)+1:m+(i-1)*(m+q), i+j-1] = vtemp
        end
    end
    if p > 0
        V = V[:, p+1:end]
        y = y[:, p+1:end]
    end
    if λ > 0
        Ȳ = estimator([y zeros(size(y, 1), size(V, 1))], [V λ * I])
    else
        Ȳ = estimator(y, V)
    end

    D = Ȳ[:, 1:m] # Feed-through term (D) is first term
    Ȳ1, Ȳ2 = similar(Ȳ, q, m, l), similar(Ȳ, q, q, l)
    Y = similar(Ȳ, q, m, l)
    inds1 = (1:m) .+ m
    inds2 = (1:q) .+ 2m
    for i = 1:l
        Ȳ1[:, :, i] = Ȳ[:, inds1]
        Ȳ2[:, :, i] = Ȳ[:, inds2]
        inds1 = inds1 .+ (m+q)
        inds2 = inds2 .+ (m+q)
    end
    Y[:, :, 1] = Ȳ1[:, :, 1] + Ȳ2[:, :, 1] * D
    for k = 2:l
        Y[:, :, k] = Ȳ1[:, :, k] + Ȳ2[:, :, k] * D
        for i = 1:k-1
            Y[:, :, k] = Y[:, :, k] + Ȳ2[:, :, i] * Y[:, :, k-i]
        end
    end
    H = similar(D, size(D)..., l + 1)
    H[:, :, 1] = D
    for k = 2:l+1
        H[:, :, k] = Y[:, :, k-1]
    end
    H
end
