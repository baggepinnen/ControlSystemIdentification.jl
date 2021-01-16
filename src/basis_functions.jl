using ControlSystems
using ControlSystems: AbstractStateSpace, ssdata, ninputs, noutputs


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

function kautz(a::AbstractVector, h)
    if maximum(abs, a) > 1
        if all(>(0) ∘ real, a)
            a = -a
        end
        a = exp.(h .* a)
        @assert all(<(1) ∘ abs, a) "Ustable poles does not make sense"
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


function laguerre_oo(a::Number, Nq)
    A = diagm(fill(-a, Nq))
    for i = 2:Nq
        A[diagind(A, i-1)] .+= 2a*(-1)^i
    end
    C = sqrt(2a)*(-1).^(0:Nq-1)'
    ss(A,I(Nq),C,0)
end


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


function ControlSystems.freqresp(P::LTISystem, basis::SomeBasis, ω::AbstractVector, p)
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
        vec.(eachslice(Gs, dims=3))
    else
        vec.(eachslice(Gs, dims=2))
    end
end

function filter_bank(basis::AbstractStateSpace{<:Discrete}, signal::AbstractVector)
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


Base.one(::TransferFunction{Continuous, ControlSystems.SisoRational{Float64}}) = tf(1)


reflect(x) = complex(-abs(real(x)), imag(x))
function minimum_phase(G::TransferFunction)
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
    G = ControlSystems.balance_statespace(G)[1]
    try
        G = balreal(G)[1]
    catch
    end
    G
end