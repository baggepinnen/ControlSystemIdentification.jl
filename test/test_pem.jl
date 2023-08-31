using ControlSystemIdentification, Optim, ControlSystemsBase
using ControlSystemsBase.DemoSystems: resonant
Random.seed!(1)
T = 1000
sys = c2d(resonant(ω0 = 0.1) * tf(1, [0.1, 1]), 1)# generate_system(nx, nu, ny)
nx = sys.nx
nu = 1
ny = 1
x0 = zeros(nx)
sim(sys, u, x0 = x0) = lsim(sys, u, 1:T, x0 = x0)[1]
sysn = c2d(resonant(ω0 = 1) * tf(1, [0.1, 1]), 1)

σu = 1e-6
σy = 1e-6

u  = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d  = iddata(yn, un, 1)

sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, show_every=100, safe=true)
@test freqresptest(sys, sysh.sys) < 1e-2
@test Optim.minimum(opt) < T*1e-4

# test longer prediction horizon
sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, show_every=100, h=2, safe=true)
@test freqresptest(sys, sysh.sys) < 1
@test Optim.minimum(opt) < T*1e-4

predplot(sysh, d; h=10)


# Test with output nonlinearity
ynn = abs.(yn)
dn  = iddata(ynn, un, 1)
output_nonlinearity = (y) -> y .= abs.(y)

for i = 1:10
    sysh, x0h, opt = ControlSystemIdentification.newpem(dn, nx; show_every=500, safe=true, output_nonlinearity)
    if freqresptest(sys, sysh.sys) < 1e-2 && Optim.minimum(opt) < T*1e-4
        @test true
        break
    end
    i == 10 && @test false
end


# Test with some noise
# Only measurement noise
σu = 0.0
σy = 0.1
u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
# sysh, x0h, opt = pem(d, nx = nx, focus = :prediction)
sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, focus = :prediction, show_every=1000, safe=true)
@test Optim.minimum(opt) < T*2σy^2 * T # A factor of 2 margin

# Only input noise
σu = 0.1
σy = 0.0
u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, focus = :prediction, show_every=1000, safe=true)
@test Optim.minimum(opt) < T*1e-3 # Should depend on system gramian, but too lazy to figure out
@test freqresptest(sys, sysh.sys) < 0.5

## Both noises
σu = 0.02
σy = 0.02

u = 10randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, u, 1)
sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx+1, focus = :simulation, show_every=100, iterations=3000, h=10, safe=true)

# yh = predict(sysh, d; h=100)
yh = simulate(sysh, d)
@test yh ≈ predict(sysh, d) # since we estiamted using simulation focus there is no predictor
fit = ControlSystemIdentification.modelfit(y, yh)[]
@test fit > 50
# plot([y' yh'], label=fit)
#

@test Optim.minimum(opt) < T*2σy^2  # A factor of 2 margin
@test norm(sys - sysh.sys) < 1e-1

##

# Simulation error minimization
σu = 0.01
σy = 0.01

u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
# @time sysh, x0h, opt = pem(d, nx = nx, focus = :simulation)
@time sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, focus = :simulation, show_every=100, safe=true)
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.3
@test Optim.minimum(opt) < T*0.01
# @test freqresptest(sys, sysh) < 1e-1


## MIMO
@testset "MIMO" begin
    Random.seed!(0)
    G = DemoSystems.doylesat()
    T = 1000
    Ts = 0.01
    sys = c2d(G, Ts)
    nx = sys.nx
    nu = sys.nu
    ny = sys.ny
    x0 = zeros(nx)
    sim(sys, u, x0 = x0) = lsim(sys, u; x0)[1]

    σy = 1e-1

    u  = randn(nu, T)
    y  = sim(sys, u, x0)
    yn = y .+ σy .* randn.()
    d  = iddata(yn, u, Ts)

    sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, show_every=100, safe=true)
    @test iszero(sysh.D)
    @test Optim.minimum(opt) < 30
    @test freqresptest(sys, sysh) < 0.4

    # bodeplot([sys, sysh])
    # predplot(sysh, d, x0h)

    yh = predict(sysh, d, x0h)
    @test mean(ControlSystemIdentification.nrmse(y, yh)) > 97


    # Non-zero D

    sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, show_every=10, zeroD = false, safe=true)
    if opt isa Optim.OptimizationResults
        @test Optim.minimum(opt) < 30
    end
    @test !iszero(sysh.D)
    @test freqresptest(sys, sysh) < 0.4

    # bodeplot([sys, sysh])
    # predplot(sysh, d, x0h)

    yh = predict(sysh, d, x0h)
    @test mean(ControlSystemIdentification.nrmse(y, yh)) > 96

end
# @error("Add tests for ny,nu > 1, also for all different keyword arguments")


@testset "Innovation model" begin
    @info "Testing Innovation model"
    ## Simulate data from an innovation model ans see if estimation recovers the true Kalman gain up to a similarity transform
    G = ssrand(2,2,3, Ts=1)
    K = kalman(G, I(G.nx), I(G.ny))
    sys = add_input(G, K, I(G.ny))

    T = 10000
    u = randn(2, T)
    e = 0.1randn(2, T)
    y = lsim(sys, [u; e]).y
    d = iddata(y, u, G.Ts)
    Gh = subspaceid(d, G.nx)
    Tr = find_similarity_transform(Gh, G)
    Gh2 = similarity_transform(Gh, Tr) # This should transform the model to the same coordinates as the true system

    @test Gh2.A ≈ G.A atol=0.2 rtol=0.2
    @test Gh2.K ≈ K atol=0.2 rtol=0.2

    Gh3,_ = ControlSystemIdentification.newpem(d, G.nx; sys0=Gh2, zeroD=false)
    Tr = find_similarity_transform(Gh3, G)
    Gh4 = similarity_transform(Gh3, Tr)

    @test Gh4.A ≈ G.A atol=0.2 rtol=0.2
    @test Gh4.K ≈ K atol=0.2 rtol=0.2
    @test_skip norm(Gh4.K-K) < norm(Gh2.K-K) # This works about 4/5
end

##
@testset "old PEM" begin
    @info "Testing old PEM"



using ControlSystemIdentification, Optim
using ControlSystemsBase.DemoSystems: resonant
Random.seed!(1)
T = 1000
sys = c2d(resonant(ω0 = 0.1) * tf(1, [0.1, 1]), 1)# generate_system(nx, nu, ny)
nx = sys.nx
nu = 1
ny = 1
x0 = zeros(nx)
sim(sys, u, x0 = x0) = lsim(sys, u, 1:T, x0 = x0)[1]
sysn = c2d(resonant(ω0 = 1) * tf(1, [0.1, 1]), 1)

σu = 1e-6
σy = 1e-6

u  = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d  = iddata(yn, un, 1)

# using BenchmarkTools
# @btime begin
# Random.seed!(0)
sysh, x0h, opt = pem(d, nx = nx, focus = :prediction, iterations=5000)
# sysh, opt = ControlSystemIdentification.newpem(d, nx, optimizer=LBFGS(), safe=true)
# bodeplot([sys,ss(sysh)], exp10.(range(-3, stop=log10(pi), length=150)), legend=false)
# end
# 462ms 121 29
# 296ms
# 283ms
# 173ms
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
# @test freqresptest(sys, ss(sysh)) < 1e-2
# yh = sim(sysh, u, x0h)
@test Optim.minimum(opt) < 1e-2

# Test with some noise
# Only measurement noise
σu = 0.0
σy = 0.1
u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
sysh, x0h, opt = pem(d, nx = nx, focus = :prediction)
# sysh, opt = ControlSystemIdentification.newpem(d, nx, focus = :prediction, safe=true)
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test Optim.minimum(opt) < 2σy^2 * T # A factor of 2 margin
# @test freqresptest(sys, sysh) < 1e-1

# Only input noise
σu = 0.1
σy = 0.0
u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
@time sysh, x0h, opt = pem(d, nx = nx, focus = :prediction)
# sysh, opt = ControlSystemIdentification.newpem(d, nx, focus = :prediction, safe=true)
@test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test Optim.minimum(opt) < 1e-3*T # Should depend on system gramian, but too lazy to figure out

end
