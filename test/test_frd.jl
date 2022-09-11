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
@test k.r[end-1] > 0.5
i = findfirst(k.w .> ωn)
@test mean(k.r[i .+ (-2:5)]) < 0.6
G, N = tfest(dn, 0.02)
@test ControlSystemsBase.issiso(G) 
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
P, C = tf(1.0, [1, 1]), pid(1, 1)
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



## Algebra
G = tf(1, [1, 2])
w = exp10.(LinRange(-2, 2, 20))
Gf = FRD(w, G)

@test feedback(Gf, 1) ≈ FRD(w, feedback(G, 1))
@test feedback(1, Gf) ≈ FRD(w, feedback(1, G))

@test feedback(Gf, Gf) ≈ FRD(w, feedback(G, G))

@test feedback(Gf, G) ≈ FRD(w, feedback(G, G))
@test feedback(G, Gf) ≈ FRD(w, feedback(G, G))


## Indexing
Gf[0.011rad:0.03rad] == Gf[2:3]
Gf[(0.011/2pi)Hz:(0.03/2pi)Hz] == Gf[2:3]


## MIMO
G = ss(tf(1, [1,1,1]))
G = let (A,B,C,D) = ssdata(G)
    ss(A,B,I(2),0)
end
G = c2d(G, 0.01)


T = 1000
u = randn(1,T)
res = lsim(G, u)
d = iddata(res)
H,N = tfest(d)
@test size(H.r,1) == 2
@test size(H.r,2) == 1
@test size(H.r,3) == T

@test !ControlSystemsBase.issiso(H)
plot(H, plotphase=true)

@test H.Ts == G.Ts
H5 = H[1:5]
@test H5.w == H.w[1:5]
@test H5.r == H.r[:,:,1:5]


## MIMO
ny,nu,nx = 2,3,4                     # number of outputs, inputs and states
Ts = 1                                  # Sample time
G = ssrand(ny,nu,nx; Ts, proper=true);  # Generate a random system

N = 200                                 # Number of frequency points
w = range(0, stop=pi/Ts-1/N, length=N)  # Frequency vector

frd = FRD(w, G);                        # Build a frequency-response data object
@test frd.r == freqresp(G, w)
@test frd.w == w

# Tests for FRD printing
littlefrd = FRD([1, 2], [im, 2*im])
@test_nowarn show(littlefrd)