using ControlSystemIdentification, ControlSystems, Statistics, Random
Random.seed!(0)
wtest = exp10.(LinRange(-3, log10(pi), 30))
freqresptest(G, model) = quantile(
    vec(log10.(abs2.(freqresp(model, wtest))) - log10.(abs2.(freqresp(G, wtest)))),
    0.9,
)

freqresptest(G, model, tol) = freqresptest(G, model) < tol


N = 3000
h = 0.05
t = range(0, length = N, step = h)


ω1, ω2 = 10, 20
f1, f2 = ω1 / 2π, ω2 / 2π
ωvec = exp10.(LinRange(-0, log10(pi / h), 300))


G1 = c2d(tf(ω1, [1, 2 * 0.02 * ω1, ω1^2]), h)
G2 = c2d(tf(ω2, [1, 2 * 0.02 * ω2, ω2^2]), h)
# bodeplot([G1, G2], ωvec)


e = randn(1,N)
y1, y2 = lsim(G1, e, t)[1], lsim(G2, e, t)[1]
d = iddata(y1 + y2, e, h)


# Gh = arx(d,2,2)
# bodeplot([G1, G2, Gh], ωvec)


##

# P = mt_pgram(vec(d.y), fs=1/h)

H1 = Bandstop(0.9 * f1, 1.1 * f1, fs = d.fs)
H2 = Bandstop(0.9 * f2, 1.1 * f2, fs = d.fs)


Gf1 = arx(d, 2, 2, estimator = weighted_estimator(H1))
Gf2 = arx(d, 2, 2, estimator = weighted_estimator(H2))


@test norm(poles(G1) - poles(Gf1)) / norm(poles(G1)) < 0.05
@test norm(poles(G2) - poles(Gf2)) / norm(poles(G1)) < 0.05

@test freqresptest(G1, Gf1) < 0.22
@test freqresptest(G2, Gf2) < 0.22

isinteractive() && bodeplot(
    [G1, G2, Gf1, Gf2],
    ωvec,
    plotphase = false,
    lab = ["G1" "G2" "Focus f1" "Focus f2"],
)
# plot!(2pi .* P.freq, sqrt.(P.power), l=(:black, 2, :dash), xlims=extrema(ωvec))

Random.seed!(0)
N = 2000
e = randn(1,N)
t = range(0, length = N, step = h)
y1, y2 = lsim(G1, e, t)[1], lsim(G2, e, t)[1]
d = iddata(y1 + y2, e, h)

# d.y .+= 0.1 .* randn.()


H1 = Bandstop(0.5 * f1, 1.2 * f1, fs = 1 / h)
H2 = Bandstop(0.9 * f2, 1.5 * f2, fs = 1 / h)


Gf1 = n4sid(d, 2, Wf = H1)
Gf2 = n4sid(d, 2, Wf = H2)
Gf1 = Gf1.sys
Gf2 = Gf2.sys


@test norm(poles(G1) - poles(Gf1)) / norm(poles(G1)) < 0.05
@test norm(poles(G2) - poles(Gf2)) / norm(poles(G1)) < 0.05

@test freqresptest(G1, Gf1) < 1
@test freqresptest(G2, Gf2) < 1

# isinteractive() && bodeplot([G1, G2, Gf1, Gf2], ωvec, plotphase=false, lab=["G1" "G2" "Focus f1" "Focus f2"])
