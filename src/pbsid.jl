"""
    pbsid(
        data::InputOutputData,
        nx = :auto;
        verbose = false,
        f = nx === :auto ? min(length(data) ÷ 20, 50) : 2nx + 10,  # future horizon
        p = f,  # past horizon (must be >= f)
        λ = 0.0,  # regularization parameter (Tikhonov)
        weight = 0,  # 0 = PBSIDopt, 1 = CVA-like
        zeroD = false,  # force D matrix to be zero
        stable = true,
        zeroD = false,
        svd::F1 = svd!,
        scaleU = true,
    )

Estimate a state-space model using the PBSID-opt algorithm.

# Arguments
- `data`: Identification data [`iddata`](@ref)
- `nx`: Model order (number of states). Use `:auto` for automatic selection.
- `verbose`: Print diagnostic information
- `f`: Future horizon for predictor construction
- `p`: Past horizon for VARX model (must be >= f)
- `λ`: Tikhonov regularization parameter for VARX estimation
- `weight`: Weighting scheme: 0 = PBSIDopt (default), 1 = CVA-like recursive weighting
- `zeroD`: If true, exclude direct feedthrough term D from VARX regression
- `stable`: Stabilize unstable A matrix using eigenvalue reflection
- `zeroD`: Force the D matrix to be zero in the final model
- `svd`: SVD function to use
- `scaleU`: Rescale inputs to have unit standard deviation

# Returns
`N4SIDStateSpace` containing the identified system and covariance matrices.

# References
- Chiuso, A. (2007). The role of vector autoregressive modeling in predictor-based subspace identification.
"""
function pbsid(
    d::InputOutputData,
    nx = :auto;
    verbose = false,
    f = nx === :auto ? min(length(d) ÷ 20, 50) : 2nx + 10,
    p = f,
    weight = 0,
    stable = true,
    zeroD = false,
    svd::F1 = svd!,
    scaleU = true,
) where {F1}

    nx !== :auto && f < nx && throw(ArgumentError("f must be at least nx"))
    p >= f || throw(ArgumentError("p (past horizon) must be >= f (future horizon)"))

    y, u = copy(time2(output(d))), copy(time2(input(d)))
    Ts = sampletime(d)

    if scaleU
        CU = vec(std(u, dims=2))
        CU[CU .== 0] .= 1
        u ./= CU
    end

    s, X, VARX = _pbsid_compute_x(u, y, f, p; weight, zeroD, svd)
    S = s.S

    if nx === :auto
        nx = sum(S .> sqrt(S[1] * S[end]))
        verbose && @info "Choosing order $nx"
    end

    verbose && @info "PBSIDopt: nx=$nx, f=$f, p=$p, weight=$weight"

    x = X[1:nx, :]

    A, B, C, D, K = _pbsid_mats(x, u, y, f, p; stable, verbose)


    # Return as N4SIDStateSpace
    # N4SIDStateSpace(sys, K, Q, R, S, P, x0, singular_values)
    # We'll compute approximate covariances from residuals
    N_state = size(x, 2)
    u_aligned = u[:, p+1:p+N_state]
    y_aligned = y[:, p+1:p+N_state]

    # Compute residuals for covariance estimation
    e = y_aligned - C * x - D * u_aligned
    x_pred = A * x[:, 1:end-1] + B * u_aligned[:, 1:end-1]
    w = x[:, 2:end] - x_pred

    fve = 1 - sum(abs2, e) / sum(abs2, y_aligned .- mean(y_aligned, dims=2))

    Ns = size(e, 2)
    R = (e * e') / Ns
    Q = (w * w') / (Ns - 1)
    Scov = zeros(nx, size(y, 1))  # cross-covariance (approximate as zero)

    # Compute P from DARE if possible
    P = try
        R_reg = R + 1e-10 * I
        Ptmp, _, _ = ared(A', C', Q, R_reg)
        Ptmp
    catch
        Matrix{Float64}(I(nx))  # fallback to identity
    end

    x0 = x[:, 1]  # initial state estimate

    if scaleU
        B = B ./ CU'
        D = D ./ CU'
    end

    sys = ss(A, B, C, D, Ts)

    return N4SIDStateSpace(sys, Q, R, Scov, K, P, x0, s, fve)
end



function _pbsid_compute_x(u, y, f, p; weight=0, zeroD=false, svd)
    # Get dimensions
    r, N = size(u)  # r inputs, N samples
    l = size(y, 1)  # l outputs
    m = r + l       # combined dimension

    # Check dimensions
    @assert size(y, 2) == N "u and y must have the same number of samples"
    @assert p >= f "p must be >= f"

    # Number of usable samples after forming past/future windows
    j = N - p - f + 1

    # Construct past data matrix Z (stacked [u; y] history for p steps)
    # Z has dimension (p*m) × j
    Z = zeros(p * m, j)
    for i in 1:p
        Z[(i-1)*m+1 : (i-1)*m+r, :] = u[:, p-i+1 : p-i+j]
        Z[(i-1)*m+r+1 : i*m, :] = y[:, p-i+1 : p-i+j]
    end

    # Construct future output matrix Y (stacked outputs for f steps)
    # Y has dimension (f*l) × j
    Y = zeros(f * l, j)
    for i in 1:f
        Y[(i-1)*l+1 : i*l, :] = y[:, p+i : p+i+j-1]
    end

    # If zeroD, we need to construct future input matrix and regress it out
    if zeroD
        # Construct future input matrix Uf
        Uf = zeros(f * r, j)
        for i in 1:f
            Uf[(i-1)*r+1 : i*r, :] = u[:, p+i : p+i+j-1]
        end
        Zext = [Z; Uf]
        VARXext = Y / Zext
        VARX = VARXext[:, 1:p*m]
    else
        VARX = Y / Z
    end

    LK = zeros(f * l, p * m)
    LK .= VARX

    # Apply CVA-like weighting
    if weight == 1
        # CVA: weight by inverse square root of output covariance
        Yf = Y
        Wf = Yf * Yf' / j
        Wf_inv_sqrt = inv(sqrt(Symmetric(Wf)))
        LK = Wf_inv_sqrt * LK
    end

    LKZ = LK * Z
    s = svd(LKZ)
    S = s.S

    X = Diagonal(S) * s.Vt
    return s, X, VARX
end

function _pbsid_mats(x, u, y, f, p; stable=false, verbose=false)
    nx = size(x, 1)  # state dimension
    nu = size(u, 1)  # number of inputs
    ny = size(y, 1)  # number of outputs
    N = size(x, 2)  # number of samples in state sequence

    # Align u, y with x by removing first p samples
    # x starts at time p+1, so we need u and y from p+1 onwards
    u_aligned = u[:, p+1:p+N]
    y_aligned = y[:, p+1:p+N]

    # Estimate C and D from output equation: y = C*x + D*u
    # [C D] = Y * pinv([X; U])
    XU = [x[:, 1:end-1]; u_aligned[:, 1:end-1]]
    Y_out = y_aligned[:, 1:end-1]

    CD = Y_out / XU
    C = CD[:, 1:nx]
    D = CD[:, nx+1:end]

    # Compute innovation (residual) e = y - C*x - D*u
    e = y_aligned - C * x - D * u_aligned

    # Estimate A, B, K from state equation: x(k+1) = A*x(k) + B*u(k) + K*e(k)
    # [A B K] = X(:,2:end) * pinv([X(:,1:end-1); U(:,1:end-1); e(:,1:end-1)])
    XUe = [x[:, 1:end-1]; u_aligned[:, 1:end-1]; e[:, 1:end-1]]
    X_next = x[:, 2:end]

    ABK = X_next / XUe
    A = ABK[:, 1:nx]
    B = ABK[:, nx+1:nx+nu]
    K = ABK[:, nx+nu+1:end]

    # Force A to be stable if requested
    if stable == 1
        A = stabilize_eigenvalues_method1(A)
    elseif stable == 2
        A = stabilize_eigenvalues_method2(A)
    end

    # Recompute Kalman gain K via discrete algebraic Riccati equation (DARE)
    # For the innovation form, we solve the DARE to get the steady-state K

    # Estimate noise covariances from residuals
    # Process noise covariance Q and measurement noise covariance R
    # Innovation covariance S (cross-covariance)

    # Recompute residuals with potentially modified A
    x_pred = A * x[:, 1:end-1] + B * u_aligned[:, 1:end-1]
    process_residual = x[:, 2:end] - x_pred

    # Covariance matrices
    Ns = size(e, 2)
    R = (e * e') / Ns  # measurement noise covariance
    Q = (process_residual * process_residual') / (Ns - 1)  # process noise covariance
    S_cov = (process_residual * e[:, 1:end-1]') / (Ns - 1)  # cross-covariance

    if any(abs.(eigvals(A-K*C)) .> 1.0)
        verbose && @warn "Estimated predictor is unstable, recomputing K via DARE"
        R_reg = R + 1e-10 * I
        K = kalman(Discrete, A, C, Q, R_reg)
    end

    return A, B, C, D, K
end

"""
    stabilize_eigenvalues_method1(A)

Force matrix A to be stable by reflecting unstable eigenvalues inside unit circle.
Method 1: Simple eigenvalue reflection.
"""
function stabilize_eigenvalues_method1(A)
    F = eigen(A)
    λ = F.values
    V = F.vectors

    # Reflect eigenvalues outside unit circle
    λ_stable = map(λi -> abs(λi) > 1 ? λi / abs(λi)^2 : λi, λ)

    # Reconstruct A
    A_stable = real(V * Diagonal(λ_stable) / V)
    return A_stable
end

"""
    stabilize_eigenvalues_method2(A)

Force matrix A to be stable using Schur decomposition.
Method 2: More numerically stable approach.
"""
function stabilize_eigenvalues_method2(A)
    F = schur(A)
    T = F.T
    Z = F.Z

    # Modify diagonal of T (eigenvalues) to be inside unit circle
    n = size(T, 1)
    for i in 1:n
        if abs(T[i, i]) > 1
            T[i, i] = T[i, i] / abs(T[i, i])^2
        end
    end

    # Reconstruct A
    A_stable = real(Z * T * Z')
    return A_stable
end
