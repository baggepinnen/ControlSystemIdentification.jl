using ControlSystemIdentification, ControlSystems
n = 5
w = exp10.(LinRange(-2, 3, 1000))
av = exp10.(LinRange(-2, 3, n))
signal = randn(1000)

for basis in [kautz(av), laguerre(1,n), laguerre_oo(1,n), adhocbasis(av)]
    p = rand(basislength(basis))
    F = sum_basis(basis, p)
    fr1 = freqresp(F, w) |> vec
    fr2 = freqresp(tf(1), basis, w, p) |> vec
    @show norm(fr1-fr2)
end

basis = kautz(av, 0.001)
y = filter_bank(basis, signal)
@test size(y) == (1000, n)







## Test that it's possible to identify a basis model in time domain

N = 2000
h = 0.01
t = range(0, step=h, length=N)
u = randn(N)
G = tf(0.8, [1, -0.9], 0.01)
signal = lsim(G, u, t)[1][:]

n = 10
av = exp10.(LinRange(0, log10(80pi), n))
w = exp10.(LinRange(-2, log10(100pi), 300))
basis = kautz(av, h)
Y = filter_bank(basis, u)
p = Y\signal
Gh = sum_basis(basis, p)

@test mean(abs2, freqresp(G, w) - freqresp(Gh, w)) < 0.1

if isinteractive()
    bodeplot(G, w)
    bodeplot!(Gh, w)
end
