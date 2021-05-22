import ControlSystemIdentification: InputOutputData, time1
## ==========================================================


r = 5 # the maximal prediction horizon used
s1 = 10 # number of past outputs
s2 = 10


h = 0.001
u = randn(1, 200)
sys = ss([0.9], [1], [1], 0, h)
y, t, x = lsim(sys, u)
d = iddata(y,u,h)
p = sys.ny
m = sys.nu
s = s1*p + s2*m


function n4sid2(
    data::InputOutputData,
    nx = :auto;
    verbose = true,
    r = 5, # the maximal prediction horizon used
    s1 = 10, # number of past outputs
    s2 = 10, # number of past inputs
    γ = nothing,
    W = :N4SID,
    R = :I, # :S1, :sqrt(S1)
    zeroD = false,
    svd::F1 = svd,
    estimator::F2 = \,
) where {F1,F2}

    y, u = time1(output(data)), time1(input(data))
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
    @assert size(Y) == (r*p, N)
    @assert size(U) == (r*m, N)
    φs(t) = [ # 10.114
        y[t-1:-1:t-s1, :] |> vec # QUESTION: not clear if vec here or not, Φ should be come s × N, should maybe be transpose before vec
        u[t-1:-1:t-s2, :] |> vec
    ]

    Φ = reduce(hcat, [φs(t) for t ∈ t0:t0+N-1]) # 10.108. Note, t can not start at 1 as in the book since that would access invalid indices for u/y. At time t=t0, φs(t0-1) is the first "past" value

    UΦY = [U; Φ; Y]
    l = lq(UΦY)
    L = l.L
    Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
    @assert size(Q) == (p*r+m*r+s, N) "size(Q) == $(size(Q))"
    Uinds = 1:size(U,1)
    Φinds = (1:size(Φ,1)) .+ Uinds[end]
    Yinds = (1:size(Y,1)) .+ Φinds[end]
    @assert Yinds[end] == p*r+m*r+s
    L21 = L[Φinds, Uinds]
    L22 = L[Φinds, Φinds]
    L32 = L[Yinds, Φinds]
    Q1 = Q[Uinds, :]
    Q2 = Q[Φinds, :]

    Ĝ = L32*(L22\[L21 L22])*[Q1; Q2] # this G is used for N4SID weight, but also to form Yh for all methods
    # 2. Select weighting matrices W1 (rp × rp)
    # and W2 (p*s1 + m*s2 × α) = (s × α)
    if W ∈ (:MOESP, :N4SID)
        if W === :MOESP
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ*ΠUt
            G = L32*Q2 #* 1/N# QUESTION: I added the N here
        elseif W === :N4SID
            W1 = I
            # W2 = 1/N * (Φ*ΠUt*Φ')\Φ
            G = Ĝ #* 1/N
        end
    elseif W ∈ (:IVM, :CVA)
        function proj(Yi, U)
            UY = [U; Yi]
            l = lq(UY)
            L = l.L
            Q = Matrix(l.Q) # (pr+mr+s × N) but we have adjusted effective N
            Uinds = 1:size(U,1)
            Yinds = (1:size(Yi,1)) .+ Uinds[end]
            if Yi === Y
                @assert size(Q) == (p*r+m*r, N) "size(Q) == $(size(Q))"
                @assert Yinds[end] == p*r+m*r
            end
            L22 = L[Yinds, Yinds]
            Q2 = Q[Yinds, :]
            L22*Q2
        end
        YΠUt = proj(Y, U)
        G = 1/N * YΠUt*Φ' # 10.109, pr×s
        @assert size(G) == (p*r, s)
        if W === :IVM
            W1 = pinv(sqrt(1/N * (YΠUt*Y')))
            W2 = pinv(sqrt(1/N * Φ*Φ'))
            G = W1*G*W2
        elseif W === :CVA
            ΦΠUt = proj(Φ, U)
            W1 = pinv(sqrt(1/N * (YΠUt*Y')))
            W2 = pinv(sqrt(1/N * ΦΠUt*Φ'))
            G = W1*G*W2
        end
        @assert size(W1) == (r*p, r*p)
        @assert size(W2, 1) == p*s1 + m*s2
    else
        throw(ArgumentError("Unknown choice of W"))
    end

    @assert size(G, 1) == r*p
    sv = svd(G)
    if nx === :auto
        nx = sum(sv.S .> sqrt(sv.S[1] * sv.S[end]))
        verbose && @info "Choosing order $nx"
    end
    n = nx
    U1 = sv.U[:, 1:n]
    S1 = sv.S[1:n]
    V1 = sv.V[:, 1:n]

    # 3. Select R and define Or = W1\U1*R
    if R === :I
        R = I
    elseif R === :S1
        R = Diagonal(S1)
    elseif R === :sS1
        R = Diagonal(sqrt.(S1))
    else
        throw(ArgumentError("Unknown choice of R"))
    end
    Or = W1\U1*R
    C = Or[1:p, 1:n]
    A = Or[1:p*(r-1), 1:n] \ Or[p+1:p*r, 1:n]

    # 4. Estimate B, D, x0 by linear regression

    # 5. If noise model, form Xh from (10.123) and estimate noise contributions using (10.124)
    Yh,Xh = let
        # if W === :N4SID
        # else
        # end
        svi = svd(Ĝ) # to form Yh, use N4SID weight
        U1i = svi.U[:, 1:n]
        S1i = svi.S[1:n]
        V1i = svi.V[:, 1:n]
        Yh = U1i*Diagonal(S1i)*V1i' # This expression only valid for N4SID?
        Lr = R\U1i'
        Xh = Lr*Yh
        Yh,Xh
    end

    CD = Yh[1:p, :]/[Xh; zeroD*U[1:m, :]] 
    C2 = CD[1:p, 1:n]
    D2 = CD[1:p, n+1:end]

    AB = Xh[:, 2:end]/[Xh[:, 1:end-1]; U[1:m, 1:end-1]]
    A2 = AB[1:n, 1:n]
    B2 = AB[1:n, n+1:end]
    
    fve = sum(S1) / sum(sv.S)
    verbose && @info "Fraction of variance explained: $(fve)"

    sys  = ss(A,  B2, C,  D2, d.Ts)
    sys2 = ss(A2, B2, C2, D2, d.Ts)

    (; Xh, Yh, A, C, U, U1, V1, S1, sv, l, CD, sys2, sys, A2,B2,C2,D2)

    # Estimate noise covariances from 10.124
end

# res = n4sid2(d, 1; W=:MOESP, r, s1, s2)
res = n4sid2(d, 1; W=:N4SID, r, s1, s2)
# res = n4sid2(d, 1; W=:CVA, r, s1, s2)
fb = bodeplot([sys, res.sys, res.sys2], w, ticks=:default, legend=false, lab="", hz=true)
fy = plot(d.y', layout=p)
plot!(res.Yh[1,:])
plot(fb,fy, size=(1900, 800))