
function find_BD(A,K,C,U,Y,m, zeroD=false, estimator=\, weights=nothing)
    T = eltype(A)
    nx = size(A, 1)
    p = size(C, 1)
    N = size(U, 2)
    A = A-K*C
    ε = lsim(ss(A,K,C,0,1), Y)[1] # innovation sequence
    φB = zeros(p, N, m*nx)
    for (j,k) in Iterators.product(1:nx, 1:m)
        E = zeros(nx)
        E[j] = 1
        fsys = ss(A, E, C, 0, 1)
        u = U[k:k,:]
        uf = lsim(fsys, u)[1]
        r = (k-1)*nx+j
        φB[:,:,r] = uf 
    end
    φx0 = zeros(p, N, nx)
    x0u = zeros(1, N)
    for (j,k) in Iterators.product(1:nx, 1:1)
        E = zeros(nx)
        x0 = zeros(nx); x0[j] = 1
        fsys = ss(A, E, C, 0, 1)
        uf = lsim(fsys, x0u; x0)[1]
        r = (k-1)*nx+j
        φx0[:,:,r] = uf 
    end
    if !zeroD
        φD = zeros(p, N, m*p)
        for (j,k) in Iterators.product(1:p, 1:m)
            E = zeros(p)
            E[j] = 1
            fsys = ss(E, 1)
            u = U[k:k,:]
            uf = lsim(fsys, u)[1]
            r = (k-1)*p+j
            φD[:,:,r] = uf 
        end
    end

    φ3 = zeroD ? cat(φB, φx0, dims=Val(3)) : cat(φB, φx0, φD, dims=Val(3))
    # φ4 = permutedims(φ3, (1,3,2))
    φ = reshape(φ3, p*N, :)
    if weights === nothing
        BD = estimator(φ, vec(Y .- ε))
    else
        BD = estimator(φ, vec(Y .- ε), weights)
    end
    B = copy(reshape(BD[1:m*nx], nx, m))
    x0 = BD[m*nx .+ (1:nx)]
    if zeroD
        D = zeros(T, p, m)
    else
        D = reshape(BD[end-p*m+1:end], p, m)
        B .+= K*D
    end
    B,D,x0
end

function find_BDf(A, C, U, Y, λ, zeroD, Bestimator, estimate_x0)
    nx = size(A,1)
    ny, nw = size(Y)
    nu = size(U, 1)
    if estimate_x0
        ue = [U; transpose(λ)] # Form "extended input"
        nup1 = nu + 1
    else
        ue = U
        nup1 = nu
    end

    sys0 = ss(A,I(nx),C,0) 
    F = evalfr2(sys0, λ)
    # Form kron matrices
    if zeroD      
        AA = similar(U, nw*ny, nup1*nx)
        for i in 1:nw
            r = ny*(i-1) + 1:ny*i
            for j in 1:nup1
                @views AA[r, ((j-1)nx) + 1:j*nx] .= ue[j, i] .* (F[:, :, i])
            end
        end
    else
        AA = similar(U, nw*ny, nup1*nx+nu*ny) 
        for i in 1:nw
            r = (ny*(i-1) + 1) : ny*i
            for j in 1:nup1
                @views AA[r, (j-1)nx + 1:j*nx] .= ue[j, i] .* (F[:, :, i])
            end
            for j in 1:nu
                AA[r, nup1*nx + (j-1)ny + 1:nup1*nx+ny*j] = ue[j, i] * I(ny)
            end
        end
    end
    vy = vec(Y)
    YY = [real(vy); imag(vy)]
    AAAA = [real(AA); imag(AA)]
    BD = Bestimator(AAAA, YY)
    e = YY - AAAA*BD
    B = reshape(BD[1:nx*nup1], nx, :)
    D = zeroD ? zeros(eltype(B), ny, nu) : reshape(BD[nx*nup1+1:end], ny, nu)
    if estimate_x0
        x0 = B[:, end]
        B = B[:, 1:end-1]
    else
        x0 = zeros(eltype(B), nx)
    end
    return B, D, x0, e
end

function find_CDf(A, B, U, Y, λ, x0, zeroD, Bestimator, estimate_x0)
    nx = size(A,1)
    ny, nw = size(Y)
    nu = size(U, 1)
    if estimate_x0
        Ue = [U; transpose(λ)] # Form "extended input"
        Bx0 = [B x0]
    else
        Ue = U
        Bx0 = B
    end

    sys0 = ss(A,Bx0,I(nx),0)
    F = evalfr2(sys0, λ, Ue)
    # Form kron matrices
    if zeroD      
        AA = F
    else
        AA = [F; U]
    end


    YY = [real(transpose(Y)); imag(transpose(Y))]
    AAAA = [real(AA) imag(AA)]
    CD = Bestimator(transpose(AAAA), YY) |> transpose
    e = YY - transpose(AAAA)*transpose(CD)
    C = CD[:, 1:nx]
    D = zeroD ? zeros(eltype(C), ny, nu) : CD[:, nx+1:end]
    return C, D, e
end

function proj(Yi, U)
    UY = [U; Yi]
    l = lq(UY)
    L = l.L
    Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    Uinds = 1:size(U,1)
    Yinds = (1:size(Yi,1)) .+ Uinds[end]
    # if Yi === Y
        # @assert size(Q) == (p*r+m*r, N) "size(Q) == $(size(Q))"
        # @assert Yinds[end] == p*r+m*r
    # end
    L22 = L[Yinds, Yinds]
    Q2 = Q[Yinds, :]
    L22*Q2
end

"""
    subspaceid(
        data::InputOutputData,
        nx = :auto;
        verbose = false,
        r = nx === :auto ? min(length(data) ÷ 20, 20) : nx + 10, # the maximal prediction horizon used
        s1 = r, # number of past outputs
        s2 = r, # number of past inputs
        W = :MOESP,
        zeroD = false,
        stable = true, 
        focus = :prediction,
        svd::F1 = svd!,
        scaleU = true,
        Aestimator::F2 = \\,
        Bestimator::F3 = \\,
        weights = nothing,
    ) 

Estimate a state-space model using subspace-based identification.

Ref: Ljung, Theory for the user.

# Arguments:
- `data`: Identification data [`iddata`](@ref)
- `nx`: Rank of the model (model order)
- `verbose`: Print stuff?
- `r`: Prediction horizon. The model may perform better on simulation if this is made longer, at the expense of more computation time.
- `s1`: past horizon of outputs
- `s2`: past horizon of inputs
- `W`: Weight type, choose between `:MOESP, :CVA, :N4SID, :IVM`
- `zeroD`: Force the `D` matrix to be zero.
- `stable`: Stabilize unstable system using eigenvalue reflection.
- `focus`: `:prediction` or `simulation`
- `svd`: The function to use for `svd`
- `scaleU`: Rescale the input channels to have the same energy.
- `Aestimator`: Estimator function used to estimate `A,C`.
- `Bestimator`: Estimator function used to estimate `B,D`.
- `weights`: A vector of weights can be provided if the `Bestimator` is `wls`. 

# Extended help
A more accurate prediciton model can sometimes be obtained using [`newpem`](@ref), which is also unbiased for closed-loop data (`subspaceid` is biased for closed-loop data, see example in the docs). The prediction-error method is iterative and generally more expensive than `subspaceid`, and uses this function (by default) to form the initial guess for the optimization.
"""
function subspaceid(
    data::InputOutputData,
    nx = :auto;
    verbose = false,
    r = nx === :auto ? min(length(data) ÷ 20, 20) : nx + 10, # the maximal prediction horizon used
    s1 = r, # number of past outputs
    s2 = r, # number of past inputs
    γ = nothing,
    W = :MOESP,
    zeroD = false,
    stable = true, 
    focus = :prediction,
    svd::F1 = svd!,
    scaleU = true,
    Aestimator::F2 = \,
    Bestimator::F3 = \,
    weights = nothing,
) where {F1,F2,F3}

    nx !== :auto && r < nx && throw(ArgumentError("r must be at least nx"))
    y, u = copy(time1(output(data))), copy(time1(input(data)))
    if scaleU
        CU = std(u, dims=1)
        u ./= CU
    end
    t, p = size(y, 1), size(y, 2)
    m = size(u, 2)
    t0 = max(s1,s2)+1
    s = s1*p + s2*m
    N = t - r + 1 - t0

    @views @inbounds function hankel(u::AbstractArray, t0, r)
        d = size(u, 2)
        H = zeros(eltype(u), r * d, N)
        for ri = 1:r, Ni = 1:N
            H[(ri-1)*d+1:ri*d, Ni] = u[t0+ri+Ni-2, :] # TODO: should start at t0
        end
        H
    end

    # 1. Form G  (10.103). (10.100). (10.106). (10.114). and (10.108).

    Y = hankel(y, t0, r) # these go forward in time
    U = hankel(u, t0, r) # these go forward in time
    # @assert all(!iszero, Y) # to be turned off later
    # @assert all(!iszero, U) # to be turned off later
    @assert size(Y) == (r*p, N)
    @assert size(U) == (r*m, N)
    φs(t) = [ # 10.114
        y[t-1:-1:t-s1, :] |> vec # QUESTION: not clear if vec here or not, Φ should become s × N, should maybe be transpose before vec, but does not appear to matter
        u[t-1:-1:t-s2, :] |> vec
    ]

    Φ = reduce(hcat, [φs(t) for t ∈ t0:t0+N-1]) # 10.108. Note, t can not start at 1 as in the book since that would access invalid indices for u/y. At time t=t0, φs(t0-1) is the first "past" value
    @assert size(Φ) == (s, N)

    UΦY = [U; Φ; Y]
    l = lq!(UΦY)
    L = l.L
    Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    @assert size(Q) == (p*r+m*r+s, N) "size(Q) == $(size(Q)), if this fails, you may need to lower the prediction horizon r which is currently set to $r"
    Uinds = 1:size(U,1)
    Φinds = (1:size(Φ,1)) .+ Uinds[end]
    Yinds = (1:size(Y,1)) .+ (Uinds[end]+s)
    @assert Yinds[end] == p*r+m*r+s
    L1 = L[Uinds, Uinds]
    L2 = L[s1*p+(r+s2)*m+1:end, 1:s1*p+(r+s2)*m+p]
    L21 = L[Φinds, Uinds]
    L22 = L[Φinds, Φinds]
    L32 = L[Yinds, Φinds]
    Q1 = Q[Uinds, :]
    Q2 = Q[Φinds, :]

    Ĝ = L32*(L22\[L21 L22])*[Q1; Q2] # this G is used for N4SID weight, but also to form Yh for all methods
    # 2. Select weighting matrices W1 (rp × rp)
    # and W2 (p*s1 + m*s2 × α) = (s × α)
    @assert size(Ĝ, 1) == r*p
    if W ∈ (:MOESP, :N4SID)
        if W === :MOESP
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ*ΠUt
            G = L32*Q2 #* 1/N# QUESTION: N does not appear to matter here
        elseif W === :N4SID
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ
            G = Ĝ #* 1/N
        end
    elseif W ∈ (:IVM, :CVA)
        if W === :IVM
            YΠUt = proj(Y, U)
            G = YΠUt*Φ' #* 1/N # 10.109, pr×s # N does not matter here
            @assert size(G) == (p*r, s)
            W1 = sqrt(Symmetric(pinv(1/N * (YΠUt*Y')))) |> real
            W2 = sqrt(Symmetric(pinv(1/N * Φ*Φ'))) |> real
            G = W1*G*W2
            @assert size(G, 1) == r*p
        elseif W === :CVA
            W1 = L[Yinds,[Φinds; Yinds]]
            ull1,sll1 = svd(W1)
            sll1 = Diagonal(sll1[1:r*p])
            Or,Sn = svd(pinv(sll1)*ull1'*L32)
            Or = ull1*sll1*Or
            # ΦΠUt = proj(Φ, U)
            # W1 = pinv(sqrt(1/N * (YΠUt*Y'))) |> real
            # W2 = pinv(sqrt(1/N * ΦΠUt*Φ')) |> real
            # G = W1*G*W2
        end
        # @assert size(W1) == (r*p, r*p)
        # @assert size(W2, 1) == p*s1 + m*s2
    else
        throw(ArgumentError("Unknown choice of W"))
    end

    # 3. Select R and define Or = W1\U1*R
    sv = W === :CVA ? svd(L32) : svd(G)
    if nx === :auto
        nx = sum(sv.S .> sqrt(sv.S[1] * sv.S[end]))
        verbose && @info "Choosing order $nx"
    end
    n = nx
    S1 = sv.S[1:n]
    R = Diagonal(sqrt.(S1))
    if W !== :CVA
        U1 = sv.U[:, 1:n]
        V1 = sv.V[:, 1:n]
        Or = W1\(U1*R)
    end
    
    fve = sum(S1) / sum(sv.S)
    verbose && @info "Fraction of variance explained: $(fve)"

    C = Or[1:p, 1:n]
    A = Aestimator(Or[1:p*(r-1), 1:n] , Or[p+1:p*r, 1:n])
    if !all(e->abs(e)<=1, eigvals(A))
        verbose && @info "A matrix unstable, stabilizing by reflection"
        A = reflectd(A)
    end

    P, K, Qc, Rc, Sc = find_PK(L1,L2,Or,n,p,m,r,s1,s2,A,C)

    # 4. Estimate B, D, x0 by linear regression
    B,D,x0 = find_BD(A, (focus === :prediction)*K, C, transpose(u), transpose(y), m, zeroD, Bestimator, weights)
    # TODO: iterate find C/D and find B/D a couple of times

    if scaleU
        B ./= CU
    end

    # 5. If noise model, form Xh from (10.123) and estimate noise contributions using (10.124)
    # Yh,Xh = let
    #     # if W === :N4SID
    #     # else
    #     # end
    #     svi = svd(Ĝ) # to form Yh, use N4SID weight
    #     U1i = svi.U[:, 1:n]
    #     S1i = svi.S[1:n]
    #     V1i = svi.V[:, 1:n]
    #     Yh = U1i*Diagonal(S1i)*V1i' # This expression only valid for N4SID?
    #     Lr = R\U1i'
    #     Xh = Lr*Yh
    #     Yh,Xh
    # end


    # CD = Yh[1:p, :]/[Xh; !zeroD*U[1:m, :]]  
    # C2 = CD[1:p, 1:n]
    # D2 = CD[1:p, n+1:end]
    # AB = Xh[:, 2:end]/[Xh[:, 1:end-1]; U[1:m, 1:end-1]]
    # A2 = AB[1:n, 1:n]
    # B2 = AB[1:n, n+1:end]
    
    N4SIDStateSpace(ss(A,  B,  C,  D, data.Ts), Qc,Rc,Sc,K,P,x0,sv,fve)
end

"""
    subspaceid(frd::FRD, args...; estimate_x0 = false, kwargs...)

If a frequency-reponse data object is supplied
- The FRD will be automatically converted to an [`InputOutputFreqData`](@ref)
- `estimate_x0` is by default set to 0.
"""
function subspaceid(frd::FRD, Ts::Real, args...; estimate_x0 = false, weights = nothing, kwargs...)
    if weights !== nothing && ndims(frd.r) > 1
        nu = size(frd.r, 2)
        weights = repeat(weights, nu)
    end
    data = ifreqresp(frd)
    subspaceid(data, Ts, args...; weights, estimate_x0, kwargs...)
end

"""
    subspaceid(data::InputOutputFreqData,
        Ts = data.Ts,
        nx = :auto;
        cont = false,
        verbose = false,
        r = nx === :auto ? min(length(data) ÷ 20, 20) : 2nx, # Internal model order
        zeroD = false,
        estimate_x0 = true,
        stable = true, 
        svd = svd!,
        Aestimator = \\,
        Bestimator = \\,
        weights = nothing
    )

Estimate a state-space model using subspace-based identification in the frequency domain.

# Arguments:
- `data`: A frequency-domain identification data object.
- `Ts`: Sample time at which the data was collected
- `nx`: Desired model order, an interer or `:auto`.
- `cont`: Return a continuous-time model? A bilinear transformation is used to convert the estimated discrete-time model, see function `d2c`.
- `verbose`: Print stuff?
- `r`: Internal model order, must be ≥ `nx`.
- `zeroD`: Force the `D` matrix to be zero.
- `estimate_x0`: Esimation of extra parameters to account for initial conditions. This may be required if the data comes from the fft of time-domain data, but may not be required if the data is collected using frequency-response analysis with exactly periodic input and proper handling of transients.
- `stable`: For the model to be stable (uses [`schur_stab`](@ref)).
- `svd`: The `svd` function to use.
- `Aestimator`: The estimator of the `A` matrix (and initial `C`-matrix).
- `Bestimator`: The estimator of B/D and C/D matrices.
- `weights`: An optional vector of frequency weights of the same length as the number of frequencies in `data.
"""
function subspaceid(
    data::InputOutputFreqData,
    Ts::Real = data.Ts,
    nx::Union{Int, Symbol} = :auto;
    cont = false,
    verbose = false,
    r = nx === :auto ? min(length(data) ÷ 20, 20) : 2nx, # the maximal prediction horizon used
    zeroD = false,
    estimate_x0 = true,
    stable = true, 
    svd::F1 = svd!,
    Aestimator::F2 = \,
    Bestimator::F3 = \,
    weights = nothing,
) where {F1,F2,F3}

    w = data.w
    Ts ≤ 2π/maximum(w) || error("Highest frequency ($(maximum(w))) is larger that the Nyquist frequency for sample time Ts = $Ts")
    nx !== :auto && r < nx && throw(ArgumentError("r must be at least nx"))
    y, u = time2(output(data)), time2(input(data))

    ny, nw = size(y)
    
    λ = cis.(w)

    if weights !== nothing
        W = Diagonal(weights)
        y = y*W
        u = u*W
        # λ = W*λ # Verified to not be needed
    end

    ue = estimate_x0 ? [u; transpose(λ)] : u
    nu = size(ue, 1)
    U = zeroD ? similar(u, (r-1)nu, nw) : similar(u, r*nu, nw)
    Y = similar(y, r*ny, nw)

    @views for i in 1:nw
        U[1:nu, i] = ue[:, i]
        Y[1:ny, i] = y[:, i]
        λi = λ[i]

        for j in 2:r
            Y[((j-1)ny + 1:j*ny), i] = λi*y[:, i]
            if !zeroD || j < r
                U[((j-1)nu + 1):j*nu, i] = λi*ue[:, i]
            end
            λi *= λ[i]
        end
    end

    AA = [real(U) imag(U); real(Y) imag(Y)]
    L = lq!(AA).L
    if zeroD
        L22 = L[((r-1)nu + 1):end, ((r-1)nu + 1):end]
    else
        L22 = L[(r*nu + 1):end, (r*nu + 1):end]
    end
    
    S = svd(L22)
    if nx === :auto
        nx = sum(sv.S .> sqrt(sv.S[1] * sv.S[end]))
        verbose && @info "Choosing order $nx"
    end

    # Observability matrix given by U, C is the first block-row
    Or = S.U 
    C = Or[1:ny, 1:nx]
    A = Aestimator(Or[1:(r-1)ny, 1:nx], Or[ny+1:ny*r, 1:nx])
    if stable && !all(e->abs(e)<=1, eigvals(A))
        verbose && @info "A matrix unstable, stabilizing by Schur projection"
        A = schur_stab(A)
    end

    B,D,x0 = find_BDf(A, C, u, y, λ, zeroD, Bestimator, estimate_x0)
    C,D = find_CDf(A, B, u, y, λ, x0, zeroD, Bestimator, estimate_x0)
    B,D,x0 = find_BDf(A, C, u, y, λ, zeroD, Bestimator, estimate_x0)
    C,D = find_CDf(A, B, u, y, λ, x0, zeroD, Bestimator, estimate_x0)
    sysd = ss(A,B,C,D, Ts)
    sys = cont ? d2c(sysd, :tustin) : sysd
    sys, x0
end


function find_PK(L1,L2,Or,n,p,m,r,s1,s2,A,C)
    X1 = L2[p+1:r*p, 1:m*(s2+r)+p*s1+p]
    X2 = [L2[1:r*p,1:m*(s2+r)+p*s1] zeros(r*p,p)]
    vl = [Or[1:(r-1)*p, 1:n]\X1; L2[1:p, 1:m*(s2+r)+p*s1+p]]
    hl = [Or[:,1:n]\X2 ; [L1 zeros(m*r,(m*s2+p*s1)+p)]]
    
    K0 = vl*pinv(hl)
    W = (vl - K0*hl)*(vl-K0*hl)'
    
    Q = W[1:n,1:n] |> Hermitian
    S = W[1:n,n+1:n+p]
    R = W[n+1:n+p,n+1:n+p] |> Hermitian
    
    local P, K
    try
        a = 1/sqrt(mean(abs, Q)*mean(abs, R)) # scaling for better numerics in ared
        P, _, Kt, _ = ControlSystemIdentification.MatrixEquations.ared(copy(A'), copy(C'), a*R, a*Q, a*S)
        K = Kt' |> copy
    catch e
        @error "Failed to estimate kalman gain, got error" e
        P = I(n)
        K = zeros(n, p)
    end
    P, K, Q, R, S
end

function reflectd(x)
    a = abs(x)
    a < 1 && return oftype(cis(angle(x)),x)
    1/a * cis(angle(x))
end

function reflectd(A::AbstractMatrix)
    D,V = eigen(A)
    D = reflectd.(D)
    A2 = V*Diagonal(D)/V
    if eltype(A) <: Real
        return real(A2)
    end
    A2
end

"""
    schur_stab(A::AbstractMatrix{T}, ϵ = 0.01)

Stabilize the eigenvalues of discrete-time matrix `A` by transforming `A` to complex
Schur form and projecting unstable eigenvalues 1-ϵ < λ ≤ 2 into the unit disc.
Eigenvalues > 2 are set to 0.
"""
function schur_stab(A::AbstractMatrix{T}, ϵ=0.01) where T
    S = schur(complex.(A))
    for i in diagind(A)
        λ = S.T[i]
        aλ = abs(λ)
        if 1 < aλ ≤ 2
            λ = λ*(2/aλ - 1)
        elseif aλ > 2
            λ = complex(0.0, 0.0)
        elseif 1-ϵ < aλ ≤ 1
            λ = (1-ϵ)*cis(angle(λ))
        end
        S.T[i] = λ
    end
    A2 = S.Z*S.T*S.Z'
    T <: Real ? real(A2) : A2
end

# plotly(show=false)
# ## ==========================================================

# h = 0.05
# w = 2pi .* exp10.(LinRange(-2, log10(1/2h), 500))
# u = randn(1, 2000)
# # sys = ss([0.9], [1], [1], 0, h)
# # sys = ss([0.98 0.1; 0 0.89], [0,1], [1 0], 1, h)
# # sys = c2d(ss(tf(1^2, [1, 2*1*0.1, 1^2])), h)
# sys = ssrand(2,1,5, Ts=h)
# y, t, x = lsim(sys, u)
# yn = y .+ 1 .* randn.()


# d = iddata(yn,u,h)
# p = sys.ny
# m = sys.nu
# nx = sys.nx


# r = 30 # the maximal prediction horizon used
# s1 = 10 # number of past outputs
# s2 = 10
# s = s1*p + s2*m


# # res = subspaceid(d, nx; W=:MOESP, r, s1, s2)
# # res = subspaceid(d, nx; W=:N4SID, r, s1, s2)
# # res = subspaceid(d, nx; W=:CVA, r, s1, s2)
# # res = subspaceid(d, nx; W=:IVM, r, s1, s2)
# # fb = bodeplot([sys, res.sys, res.sys2], w, ticks=:default, legend=false, lab="", hz=true)
# # fy = plot([y' d.y'], layout=p)
# # plot!([zeros(max(s1,s2)); res.Yh[1,:]])
# # plot(fb,fy, size=(1900, 800))

# #
# # @error "Initial state not estimated. This is probably why estimation is a bit off for real data but not for synthetic that starts at x0=0"
# bodeplot(sys, w, lab="True", plotphase=false, size=(1800, 800))
# alg = :MOESP
# for alg in [:MOESP, :N4SID, :IVM, :CVA]
#     sysh = subspaceid(d, nx; r, s1, s2, zeroD=false, verbose=true, W=alg)# , Wf = Highpass(18, fs=datav.fs) )
#     bodeplot!(sysh.sys, w, lab="$alg sys1", plotphase=false)
#     # bodeplot!([sysh.sys, sysh.sys2], w, lab=["$alg sys1" "$alg sys2"], plotphase=false)
# end
# display(current())
# ##
"""
    find_similarity_transform(sys1, sys2)

Find T such that `ControlSystems.similarity_transform(sys1, T) == sys2`

Ref: Minimal state-space realization in linear system theory: an overview, B. De Schutter

```jldoctest
julia> T = randn(3,3);

julia> sys1 = ssrand(1,1,3);

julia> sys2 = ControlSystems.similarity_transform(sys1, T);


julia> T2 = find_similarity_transform(sys1, sys2);

julia> T2 ≈ T
true
```
"""
function find_similarity_transform(sys1, sys2, method = :obsv)
    if method === :obsv
        O1 = obsv(sys1)
        O2 = obsv(sys2)
        return O1\O2
    elseif method === :ctrb
        C1 = ctrb(sys1)
        C2 = ctrb(sys2)
        return C1/C2
    else
        error("Unknown method $method")
    end
end

function evalfr2(sys::AbstractStateSpace, w_vec::AbstractVector{Complex{W}}, u) where W
    ny, nu = size(sys)
    T = promote_type(Complex{real(eltype(sys.A))}, Complex{W})
    F = hessenberg(sys.A)
    Q = Matrix(F.Q)
    A = F.H
    C = sys.C*Q
    B = Q\sys.B 
    D = sys.D
    te = sys.timeevol
    R = Array{T, 2}(undef, ny, length(w_vec))
    Bi = B*u[:, 1]
    Bc = similar(Bi, T) # for storage
    for i in eachindex(w_vec)
        @views mul!(Bi, B, u[:, i])
        Ri = @views R[:,i]
        Ri .= 0
        isinf(w_vec[i]) && continue
        copyto!(Bc,Bi) # initialize storage to Bi
        w = -w_vec[i] # This differs from standard freqresp, hence the name evalfr2
        ldiv!(A, Bc, shift = w) # B += (A - w*I)\B # solve (A-wI)X = B, storing result in B
        mul!(Ri, C, Bc, -1, 1) # use of 5-arg mul to subtract from D already in Ri. - rather than + since (A - w*I) instead of (w*I - A)
    end
    R
end

function evalfr2(sys::AbstractStateSpace, w_vec::AbstractVector{Complex{W}}) where W <: Real
    ny, nu = size(sys)
    T = promote_type(Complex{real(eltype(sys.A))}, Complex{W})
    F = hessenberg(sys.A)
    Q = Matrix(F.Q)
    A = F.H
    C = sys.C*Q
    B = Q\sys.B 
    D = sys.D
    te = sys.timeevol
    R = Array{T, 3}(undef, ny, nu, length(w_vec))
    Bc = similar(B, T) # for storage
    for i in eachindex(w_vec)
        Ri = @views R[:,:,i]
        copyto!(Ri,D) # start with the D-matrix
        isinf(w_vec[i]) && continue
        copyto!(Bc,B) # initialize storage to B
        w = -w_vec[i] # This differs from standard freqresp, hence the name evalfr2
        ldiv!(A, Bc, shift = w) # B += (A - w*I)\B # solve (A-wI)X = B, storing result in B
        mul!(Ri, C, Bc, -1, 1) # use of 5-arg mul to subtract from D already in Ri. - rather than + since (A - w*I) instead of (w*I - A)
    end
    R
end


"""
    U,Y,Ω = ifreqresp(F, ω, Ts=0)

Given a frequency response array `F: ny × nu × nω`, return input-output frequency data data consistent with `F` and an extended frequency vector `Ω` of matching length.
If `Ts > 0` is provided, a bilinear transform from continuous to discrete domain is performed on the frequency vector. This is required for subspace-based identification if the data is obtained by, e.g., frequency-response analysis.
"""
function ifreqresp(F, ω, Ts=0)
    F isa PermutedDimsArray && (F = F.parent)
    if ndims(F) == 3
        ny,nu,nw = size(F)
    else
        nw = length(F)
        ny = nu = 1
    end
    U = similar(F, nu, nw*nu)
    Y = similar(F, ny, nw*nu)
    Ω = Vector{Float64}(undef, nw*nu)
    B = I(nu)

    for i in 1:nu
        r = (i-1)*nw+1:i*nw
        Y[:, r] = F[:, i, :]
        U[:, r] = repeat(B[:, i], 1, nw)
        Ω[r] = ω
    end

    if Ts > 0
        Ω = c2d(Ω, Ts)
    end

    return Y, U, Ω
end

ifreqresp(frd::FRD, Ts=0) = InputOutputFreqData(ifreqresp(frd.r, frd.w, Ts)...)