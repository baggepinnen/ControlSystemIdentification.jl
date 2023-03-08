using ControlSystemIdentification, ControlSystemsBase
n = 5
w = exp10.(LinRange(-2, 3, 1000))
av = exp10.(LinRange(-2, 3, n))
signal = randn(1,1000)

for basis in [kautz(av), laguerre(1,n), laguerre_oo(1,n), adhocbasis(av)]
    p = rand(basislength(basis))
    F = sum_basis(basis, p)
    fr1 = freqresp(F, w) |> vec
    fr2 = freqresp(tf(1), basis, w, p) |> vec
    @test norm(fr1-fr2) < 1e-10
    @test basislength(basis) == n
end
basis = laguerre_oo(1,n)
b2 = add_poles(basis, [-2])
@test basislength(b2) == n+1
@test -2 ∈ poles(b2)

basis = kautz(av, 0.001)
y = filter_bank(basis, signal)
@test size(y) == (n, 1000)







## Test that it's possible to identify a basis model in time domain

N = 2000
h = 0.01
t = range(0, step=h, length=N)
u = randn(1,N)
G = tf(0.8, [1, -0.9], 0.01)
signal = lsim(G, u, t)[1]

n = 10
av = exp10.(LinRange(0, log10(80pi), n))
w = exp10.(LinRange(-2, log10(100pi), 300))
# basis = laguerre_oo(1, n)
basis = kautz(av, h)
Y = filter_bank(basis, u)
p = Y'\vec(signal)
@test norm(signal' - Y'*p)/norm(signal) < 1e-3
Gh = sum_basis(basis, p)
@test isdiscrete(Gh) == isdiscrete(basis)

@test mean(abs2, abs2.(freqresp(G, w)) - abs2.(freqresp(Gh, w))) < 0.1

@test mean(abs2, (freqresp(G, w)) - (freqresp(Gh, w))) < 0.1

if isinteractive()
    bodeplot(G, w)
    bodeplot!(Gh, w)
end


## laguerre_id
n = 4
Ts = G.Ts
a = 10
basis = ControlSystemIdentification.laguerre_id(a, n, 0.01)
Y = filter_bank(basis, u)
p = Y'\vec(signal)
Gh = sum_basis(basis, p)
@test norm(signal' - Y'*p)/norm(signal) < 1e-3
@test isdiscrete(Gh) == isdiscrete(basis)
@test mean(abs2, abs2.(freqresp(G, w)) - abs2.(freqresp(Gh, w))) < 0.1
@test mean(abs2, (freqresp(G, w)) - (freqresp(Gh, w))) < 0.1

if isinteractive()
    bodeplot(G, w)
    bodeplot!(Gh, w)
end

