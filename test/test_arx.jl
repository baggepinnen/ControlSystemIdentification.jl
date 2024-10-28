using ControlSystemIdentification, MonteCarloMeasurements, ControlSystemsBase, Test
unsafe_comparisons(true)
wtest = exp10.(LinRange(-3, log10(pi), 30))
freqresptest(G, model) =
    maximum(abs, log10.(abs2.(freqresp(model, wtest))) .- log10.(abs2.(freqresp(G, wtest))))

freqresptest(G, model, tol) = freqresptest(G, model) < tol

"""
Compare a tf to a tf based on particles .. maybe replaye by correct dispatched isapprox
"""
function compareTFs(tf, tfStochastic, atol = 10e-5)
    aok = bok = false
    try
        bok = isapprox(numvec(tf), pmean.(numvec(tfStochastic)), atol = atol)
        aok = isapprox(denvec(tf), pmean.(denvec(tfStochastic)), atol = atol)
    catch
        return false
    end  
    return aok && bok
end

N = 20
t = 1:N
u = randn(1,N)
G = tf(0.8, [1, -0.9], 1)
y = lsim(G, u, t)[1][:]

pars = ControlSystemIdentification.params(G)
@test pars == ([0.9, 0.8], [0.9], [[0.8]], [1])
@test ControlSystemIdentification.params2poly(pars[1], 1, 1) == ([1, -0.9], [[0.8]])

na, nb = 1, 1
yr, A = getARXregressor(y, u', na, nb)
@test length(yr) == N - na
@test size(A) == (N - na, na + nb)

@test yr == y[na+1:end]
@test A[:, 1] == y[1:end-na]
@test A[:, 2] == u[1:end-1]

na = 1
d = iddata(y, u, 1)
Gh = arx(d, na, nb)
@test Gh ≈ G # Should recover the original transfer function exactly
@test freqresptest(G, Gh, 0.0001)
ω = exp10.(range(-2, stop = 1, length = 200))
# bodeplot(G,ω)
# bodeconfidence!(Gh, Σ, ω=ω, color=:blue, lab="Est")

@test arx([d,d], na, nb) ≈ Gh rtol = 1e-10

yh1 = predict(Gh, d)
yh2 = simulate(ss(Gh), d)
@test yh1 ≈ yh2


# Test MISO estimation
u2 = randn(1,N)
G2 = [G tf(0.5, [1, -0.9], 1)]
y2 = lsim(G2, [u; u2], t)[1][:]

nb = [1, 1]
d = iddata(y2, [u; u2], 1)
Gh2 = arx(d, na, nb)
@test Gh2 ≈ G2
Gh2s = arx(d, na, nb, stochastic = true)
@test compareTFs(G2, Gh2s)
# Test na < nb
## SISO
na, nb = 1, 2
G1 =  tf([1, -2], [1, -0.5, 0], 1)
u = randn(1,N)
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
Gest = arx(d, na, nb)
@test G1 ≈ Gest

Gests = arx(d, na, nb, stochastic = true)
@test compareTFs(G1, Gests)

## MISO nb1 != nb2
na, nb = 1, [2, 3]
G2 =  tf([1, -2, 3], [1, -0.5, 0, 0], 1)
G = [G1 G2]
u1 = randn(1,N)
u2 = randn(1,N)
u = [u1; u2]
y = lsim(G, u, t)[1][:]
d = iddata(y, u, 1)
Gest = arx(d, na, nb)
@test Gest ≈ G

Gests = arx(d, na, nb, stochastic = true)
@test compareTFs(G, Gests)


# Test na = 0
na, nb = 0, 1
G1 =  tf([1], [1, 0], 1)
u = randn(1,N)
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
Gest = arx(d, na, nb)
@test Gest ≈ G1

Gests = arx(d, na, nb, stochastic = true)
@test compareTFs(G1, Gests)


# Test inputdelay
## SISO
na, nb, inputdelay = 1, 1, 2
G1 =  tf([0,1], [1, -0.5,0], 1)
u = randn(1,N)
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
Gest = arx(d, na, nb, inputdelay = inputdelay)
@test Gest ≈ G1
Gests = arx(d, na, nb, inputdelay = inputdelay, stochastic = true)
@test compareTFs(G1, Gests)


## MISO
na, nb, inputdelay = 1, [1, 1], [2, 3]
G1 =  tf([0,1], [1, -0.5,0], 1)
G2 =  tf([0,0,1], [1, -0.5,0,0], 1)
G = [G1 G2]
u1 = randn(1,N)
u2 = randn(1,N)
u = [u1; u2]
y = lsim(G, u, 1:N)[1][:]
d = iddata(y, u, 1)
Gest = arx(d, na, nb, inputdelay = inputdelay)
@test Gest ≈ G

Gests = arx(d, na, nb, inputdelay = inputdelay, stochastic =  true)
@test compareTFs(G, Gests)

# direct input
G1 =  tf([0.3, 1], [1, -0.5], 1)
u = randn(1,N)
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
na, nb, inputdelay = 1,2,0
Gest = arx(d, na, nb, inputdelay = inputdelay)
@test Gest ≈ G1

Gests = arx(d, na, nb, inputdelay = inputdelay, stochastic = true)
@test compareTFs(G1, Gests)

## with inputdelay
G1 =  tf([0.3,0, 1], [1, -0.5, 0], 1)
u = @.(sin(t') + sign(sin(2t')))
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
na, nb, inputdelay = 1,3,0
Gest = arx(d, na, nb, inputdelay = inputdelay)
@test Gest ≈ G1

Gests = arx(d, na, nb, inputdelay = inputdelay, stochastic = true)
@test compareTFs(G1, Gests)

## Multiple data sets
Gest = arx([d, d], na, nb; inputdelay)

# From https://github.com/baggepinnen/ControlSystemIdentification.jl/issues/161
N  = 20000    # Number of time steps
t  = 1:N
Δt = 1        # Sample time
u  = randn(2, N) # A random control input
G  = ssrand(1, size(u, 1), size(u, 1); Ts=Δt)
y  = lsim(G, u, t)[1][:]
d  = iddata(y, u, Δt)
na = 2
nb = [3, 3]

Gest = arx([d, d], na, nb, inputdelay=[0,0])  
@test freqresptest(G, Gest, 0.0001)