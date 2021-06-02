using ControlSystems, ControlSystemIdentification
import ControlSystemIdentification: InputOutputData, time1

function find_BD(A,K,C,U,Y,m, zeroD=false, estimator=\)
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

    φ3 = zeroD ? cat(φB, φx0, dims=3) : cat(φB, φx0, φD, dims=3)
    # φ4 = permutedims(φ3, (1,3,2))
    φ = reshape(φ3, p*N, :)
    BD = estimator(φ, vec(Y .- ε))
    B = reshape(BD[1:m*nx], nx, m)
    x0 = BD[m*nx .+ (1:nx)]
    if zeroD
        D = zeros(p, m)
    else
        D = reshape(BD[end-p*m+1:end], p, m)
        B = B + K*D
    end
    B,D,x0
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
    svd::F1 = svd,
    scaleU = true,
    Aestimator::F2 = \,
    Bestimator::F3 = \,
) where {F1,F2,F3}

    nx !== :auto && r < nx && throw(ArgumentError("r must be at least nx"))
    y, u = time1(output(data)), time1(input(data))
    if scaleU
        CU = std(u, dims=1)
        u = u ./ CU
    end
    t, p = size(y, 1), size(y, 2)
    m = size(u, 2)
    t0 = max(s1,s2)+1
    s = s1*p + s2*m
    N = t - r + 1 - t0

    function hankel(u::AbstractArray, t0, r)
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
    @assert all(!iszero, Y) # to be turned off later
    @assert all(!iszero, U) # to be turned off later
    @assert size(Y) == (r*p, N)
    @assert size(U) == (r*m, N)
    φs(t) = [ # 10.114
        y[t-1:-1:t-s1, :] |> vec # QUESTION: not clear if vec here or not, Φ should become s × N, should maybe be transpose before vec, but does not appear to matter
        u[t-1:-1:t-s2, :] |> vec
    ]

    Φ = reduce(hcat, [φs(t) for t ∈ t0:t0+N-1]) # 10.108. Note, t can not start at 1 as in the book since that would access invalid indices for u/y. At time t=t0, φs(t0-1) is the first "past" value
    @assert size(Φ) == (s, N)

    UΦY = [U; Φ; Y]
    l = lq(UΦY)
    L = l.L
    Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    @assert size(Q) == (p*r+m*r+s, N) "size(Q) == $(size(Q))"
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
            W1 = sqrt((pinv(1/N * (YΠUt*Y')))) |> real
            W2 = sqrt((pinv(1/N * Φ*Φ'))) |> real
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
        sv = svd(G)
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
    B,D,x0 = find_BD(A, (focus === :prediction)*K, C, u', y', m, zeroD, Bestimator)

    if scaleU
        B = B ./ CU
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
    
    sys  = ControlSystemIdentification.N4SIDStateSpace(ss(A,  B,  C,  D, data.Ts), Qc,Rc,Sc,K,P,x0,sv,fve)
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

