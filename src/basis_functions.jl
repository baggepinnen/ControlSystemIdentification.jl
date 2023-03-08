using ControlSystemsBase
using ControlSystemsBase: AbstractStateSpace, ssdata, ninputs, noutputs


function basis_factor(p::Number)
    (A=p, B=sqrt(1-abs2(p)), C=sqrt(1-abs2(p)), D=-p)
end

# eq (3) in https://people.kth.se/~andersb/research/BF03.pdf
# IDENTIFICATION OF RATIONAL SPECTRAL DENSITIES USING ORTHONORMAL BASIS FUNCTIONS
function expand_basis(l, s) # large, small
    A = [l.A zeros(size(l.A,1), 1); s.B*l.C s.A]
    B = [l.B; s.B*l.D]
    C = [s.D*l.C s.C]
    D = s.D*l.D
    (; A, B, C, D)
end

function kautz(a::AbstractVector)
    h = 1/(pi*maximum(a))
    b = kautz(a, h)
    d2c(b)
end

"""
    kautz(a::Vector, h)

Construct a discrete-time Kautz basis with poles at `a` amd sample time `h`.
"""
function kautz(a::AbstractVector, h)
    if maximum(abs, a) > 1
        if all(>(0) ∘ real, a)
            a = -a
        end
        a = exp.(h .* a)
        @assert all(<(1) ∘ abs, a) "Including ustable poles does not make sense"
    end
    factors = basis_factor.(a)
    Q = factors[1]
    for f in factors[2:end]
        Q = expand_basis(Q, f)
    end
    T = (Q...,)
    # A,B,C,D = real.(T)
    A,B,C,D = T
    C2 = I(size(A,1))
    ss(A,B,C2,0, h)
end


"""
    laguerre_oo(a::Number, Nq)

Construct an output orthogonalized Laguerre basis of length `Nq` with poles at `-a`.
"""
function laguerre_oo(a::Number, Nq)
    A = diagm(fill(-a, Nq))
    for i = 2:Nq
        A[diagind(A, i-1)] .+= 2a*(-1)^i
    end
    C = sqrt(2a)*(-1).^(0:Nq-1)'
    ss(A,I(Nq),C,0)
end

"""
    laguerre(a::Number, Nq)

Construct a Laguerre basis of length `Nq` with poles at `-a`.
"""
function laguerre(a, Nq)
    A = diagm(fill(-a, Nq))
    A[diagind(A, 1)] .= 1
    B = diagm(fill(1, Nq))
    for i = 1:Nq-1
        for (k,j) in enumerate(diagind(A, i))
            B[j] = (-1)^i * binomial(i+k-1, k-1)
        end
    end
    Ma = sqrt(2a)*diagm((2a).^(0:Nq-1))
    C = [1 zeros(1, Nq-1)]
    ss(A,Ma*B,C,0)
end

"""
    laguerre_id(a::Number, Nq, Ts)

Construct a discrete-time Laguerre basis of length `Nq` with poles at `-a` for system identification.

NOTE: for large `Nq`, this basis may be numerically ill-conditioned. Consider applying `balance_statespace` to the resulting basis.
"""
function laguerre_id(a, Nq, Ts)
    # Ref: Time delay integrating systems: a challenge for process control industries. A practical solution
    # Ref 2: INDUSTRIAL AUTOMATION WITH BRAINWAVE – MULTIMAX AN ADAPTIVE MODEL BASED PREDICTIVE CONTROLLER https://people.ece.ubc.ca/huzmezan/docs/MultimaxPaper1.pdf
    Nq >= 2 || error("Nq must be at least 2")
    e = exp(-a*Ts)
    τ1 = e
    τ2 = Ts + 2/a*(e - 1)
    τ3 = -Ts*e - τ2
    τ4 = sqrt(2a)*(1-τ1)/a
    diags = [
        fill(τ1, Nq),
        fill((-τ1*τ2 - τ3)/Ts, Nq-1),
        [
            fill((-1)^(i-1) * τ2^(i-2)*(τ1*τ2 + τ3)/Ts^(i-1), Nq-i+1) for i in 3:Nq
        ]...
    ]
    A = diagm(((0:-1:-Nq+1) .=> diags)...)
    B = [τ4*(-τ2*Ts) ^ (i-1) for i in 1:Nq] # TODO: not sure about the τ4 here
    C = I# diagm([Ts^i for i in 1:Nq])
    ss(A,B,C,0,Ts)
end


function adhocbasis(a)
    b = map(1:length(a)-1) do i
        ss(leadlinkat(a[i], 4, 1/2)*laglink(a[i+1], 2))
    end
    [b..., leadlinkat(a[end], 4, 1/2)]
end

function simplified_laguerre(a)
    map(enumerate(a)) do (i,ai)
        tf(1, [1.0, ai])^i
    end
end


# function Q_basis(P, K0, h, q)
#     Nq = length(q)
#     Pd = c2d(P, h, :foh)
#     # Nominal Q based on nominal controller
#     K0d, _ = balreal(c2d(K0, h, :foh))
#     Q0d, _ = balreal(feedback(K0d, Pd))
#     isstable(Q0d) || error("Sampled nominal design gives unstable closed loop")

#     # Q parametrization by finite pulse response
#     AQ1 = diagm(1=>ones(Nq-2))  # Q = K/(1+PK)
#     BQ1 = q[2:Nq]            # The other Nq-1 values of pulse response
#     CQ1 = [1 zeros(1, Nq-2)]
#     DQ1 = q[1]               # The first (k=0) value of pulse response

#     Q = Q0d + ss(AQ1, BQ1, CQ1, DQ1, h)
#     T = Pd*Q
#     S = 1-T
#     PS = P*S
#     isstable(PS) || error("PS is not stable")
#     Q,T,S,PS,Pd
# end




# Utils ===============================================================================
const SomeBasis = Union{AbstractVector{<:LTISystem}, LTISystem}


basislength(v::AbstractVector) = length(v)
function basislength(v::LTISystem)
    n = ninputs(v)
    n == 1 ? noutputs(v) : n
end

function add_poles(basis::AbstractStateSpace, sys::LTISystem)
    add_poles(basis, ss(sys).A)
end

function add_poles(basis::AbstractStateSpace, Ad)
    A,B,C,D = ssdata(basis)
    T = eltype(A)
    nx,nu,ny = basis.nx,basis.nu,basis.ny
    nn = size(Ad, 1)
    @assert size(A) == size(B) "only supports laguerre_oo basis"
    B == I || @warn("B was not identity, the B matrix will be overwritten with I")

    Ae = [A zeros(size(A,1), nn); zeros(T, nn, nx) Ad]
    # Be = [B; zeros(T, nn, nu)]
    Ce = [C ones(T, ny, nn)]
    ss(Ae,I(nn+nx),Ce,0)
end


function ControlSystemsBase.freqresp(P::LTISystem, basis::SomeBasis, ω::AbstractVector, p)
    @assert ninputs(P) == noutputs(P) == 1 "Only supports SISO"
    resp_P = freqresp(P, ω) |> vec
    basis_resp = basis_responses(basis, ω, inverse=false)
    @assert length(basis_resp) == length(p) "p does not have the same length as the basis"

    FPs = map(basis_resp) do Fi
        Fi .* resp_P
    end

    Gmat = reduce(hcat, FPs)
    Gmat*p
end

function sum_basis(basis::AbstractVector, p::AbstractVector)
    out = ss(p[1]*basis[1])
    for i in 2:length(p)
        out = out + balreal(p[i] * basis[i])[1]
    end
    out
end

function sum_basis(basis::AbstractStateSpace, p::AbstractVector)
    A,B,C,D = ssdata(basis)
    @assert all(iszero, D)
    if ninputs(basis) > 1
        isdiscrete(basis) ? ss(A,B*p,C,0,basis.Ts) : ss(A,B*p,C,0)
    else
        isdiscrete(basis) ? ss(A,B,p'C,0,basis.Ts) : ss(A,B,p'C,0)
    end
end


function basis_responses(basis::AbstractVector, ω; inverse=false)
    inverse && (basis = inv.(basis))
    Gs = freqresp.(basis, Ref(ω))
    vec.(Gs)
end

function basis_responses(basis::AbstractStateSpace, ω; inverse=false)
    inverse && (basis = inv(basis))
    Gs = freqresp(basis, ω)
    if ninputs(basis) > 1
        vec.(eachslice(Gs, dims=2))
    else
        vec.(eachslice(Gs, dims=1))
    end
end

"""
    filter_bank(basis::AbstractStateSpace{<:Discrete}, signal::AbstractMatrix)

Filter `signal` through all systems in `basis`
"""
function filter_bank(basis::AbstractStateSpace{<:Discrete}, signal::AbstractMatrix)
    size(signal, 1) == 1 || throw(ArgumentError("Only supporting 1D signals"))
    no = ninputs(basis)
    t = range(0, step=basis.Ts, length=length(signal))
    if no > 1
        y,_,_ = lsim(basis, repeat(signal, 1, no), t)
    else
        y,_,_ = lsim(basis, signal, t)
    end
    y
end

function ωζ2complex(ω, ζ)
    p = ω*cis(pi-acos(ζ))
    p
end


Base.one(::TransferFunction{Continuous, ControlSystemsBase.SisoRational{Float64}}) = tf(1)


reflect(x) = complex(-abs(real(x)), imag(x))


"""
    minimum_phase(G)

Move zeros and poles of `G` from the unstable half plane to the stable.
If `G` is a statespace system, it's converted to a transfer function first.
This can incur loss of precision.
"""
function minimum_phase(G::TransferFunction{Continuous})
    z,p,k = zpkdata(G) .|> first
    z = reflect.(z)
    p = reflect.(p)
    Gmp = tf(zpk(z,p,k))
    if sign(dcgain(G)[]) != sign(dcgain(Gmp)[])
        Gmp = -Gmp
    end
    Gmp
end

function minimum_phase(G::StateSpace)
    H = tf(G)
    G = ss(minimum_phase(H))
    G = ControlSystemsBase.balance_statespace(G)[1]
    try
        G = balreal(G)[1]
    catch
    end
    G
end