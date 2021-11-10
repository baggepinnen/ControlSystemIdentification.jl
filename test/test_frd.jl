using ControlSystemIdentification
Random.seed!(1)
##
T           = 100000
h           = 1
t           = range(0, step = h, length = T)
sim(sys, u) = lsim(c2d(sys, h), u, t)[1]
σy          = 0.5
sys         = tf(1, [1, 2 * 0.1, 0.1])
ωn          = sqrt(0.3)
sysn        = tf(σy * ωn, [1, 2 * 0.1 * ωn, ωn^2])

u  = randn(1,T)
y  = sim(sys, u)
yn = y + sim(sysn, randn(size(u)))
d  = iddata(y, u, h)
dn = iddata(yn, u, 1)

# using BenchmarkTools
# @btime begin
# Random.seed!(0)
k = coherence(d)
@test mean(k.r) > 0.98

k = coherence(dn)
@test all(k.r[1:10] .> 0.9)
@test k.r[end] .> 0.7
@test k.r[findfirst(k.w .> ωn)] < 0.6
G, N = tfest(dn, 0.02)
noisemodel = innovation_form(ss(sys), syse = ss(sysn))
noisemodel.D .*= 0
bodeplot(
    [sys, sysn],
    exp10.(range(-3, stop = log10(pi), length = 200)),
    layout    = (1, 4),
    plotphase = false,
    subplot   = [1, 2],
    size      = (3 * 800, 600),
    linecolor = :blue,
)#, ylims=(0.1,300))

coherenceplot!(dn, subplot = 3)
crosscorplot!(dn, -10:100, subplot = 4)
plot!(G, subplot = 1, lab = "G Est", alpha = 0.3, title = "Process model")
plot!(√N, subplot = 2, lab = "N Est", alpha = 0.3, title = "Noise model")


for op in (+, -, *)
    @test op(G, G) isa FRD
end



ω = exp10.(LinRange(-2, 2, 100))
P, C = tf(1.0, [1, 1]), pid(kp = 1, ki = 1)
S, D, N, T = gangoffour(P, C)
S2, D2, N2, T2 = gangoffour(FRD(ω, P), FRD(ω, C))

@test ninputs(S2) == noutputs(S2) == 1
@test size(S2) == (1, 1)
@test S2 == S2
@test FRD(ω, S) ≈ S2
@test FRD(ω, D) ≈ D2
@test FRD(ω, N) ≈ N2
@test FRD(ω, T) ≈ T2

@test FRD(ω, P)*P ≈ FRD(ω, P)*FRD(ω, P)
@test FRD(ω, P)-P ≈ FRD(ω, P)-FRD(ω, P)
@test FRD(ω, P)+P ≈ FRD(ω, P)+FRD(ω, P)