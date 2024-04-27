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
ynn = abs.(yn) .- 1
unn = un .- 1
dn  = iddata(ynn, unn, 1)[1:200]
output_nonlinearity = (y, p) -> y .= abs.(y) .- p[1]
input_nonlinearity = (u, p) -> u .= u .- p[2]

nlp = [0.9, 0.9]

using Optim.LineSearches
# optimizer = LBFGS(
#     alphaguess = LineSearches.InitialStatic(alpha = 0.001),
#     # linesearch = LineSearches.BackTracking(),
# )
optimizer = NelderMead()

regularizer = (p, P) -> 0#0.0001*sum(abs2, p)

for i = 1:30
    sysh, x0h, opt, nlph = ControlSystemIdentification.newpem(dn, nx; show_every=5000, safe=true, input_nonlinearity, output_nonlinearity, nlp, optimizer, regularizer)

    local either_or
    try
        either_or = min(hinfnorm(sys-sysh.sys)[1], hinfnorm(sys+sysh.sys)[1])
    catch
        either_or = 1e10
    end

    if either_or < 1 && Optim.minimum(opt) < T*1e-3 && abs(nlph[1] - 1) < 0.1 && abs(nlph[2] - 1) < 0.1
        @test true
        break
    end
    i == 10 && @test false
end

for i = 1:30
    sysh, x0h, opt, nlph = ControlSystemIdentification.newpem(dn, nx; show_every=5000, safe=true, input_nonlinearity, output_nonlinearity, nlp, focus=:simulation, optimizer, regularizer)

    local either_or
    try
        either_or = min(hinfnorm(sys-sysh.sys)[1], hinfnorm(sys+sysh.sys)[1])
    catch
        either_or = 1e10
    end

    if either_or < 1 && Optim.minimum(opt) < T*1e-3 && abs(nlph[1] - 1) < 0.1 && abs(nlph[2] - 1) < 0.1
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


    sysh, x0h, opt = ControlSystemIdentification.newpem(d, nx, show_every=100, safe=true, h=3)
    @test iszero(sysh.D)
    @test Optim.minimum(opt) < 30
    @test freqresptest(sys, sysh) < 0.4

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


@testset "structured_pem" begin
    @info "Testing structured_pem"
    constructor = function (p)
        Mα, Mq, Mδe, Zα, Zq, Zδe = p
        V = 30.2
        A = [
            Mq Mα
            (1+Zq / V) Zα/V
        ]
        B = [Mδe; Zδe / V]
        return ss(A, B, I, 0)
    end

    # True Parameters
    Mα_true = -83.17457223750834
    Mq_true = -4.0485957994781785
    Mδe_true = -80.71377329163104
    Zα_true = -115.91677356408941
    Zq_true = -0.573560626095269
    Zδe_true = 32.13261325199936

    Ts = 0.01
    t = 0:Ts:10

    p_true = [Mα_true, Mq_true, Mδe_true, Zα_true, Zq_true, Zδe_true]
    p0 = [Mα_true, Mq_true, Mδe_true, Zα_true, Zq_true, Zδe_true] .* 1.3

    P_true = constructor(p_true)
    u = sign.(sin.((1 / 3 * t)')) .+ 0.2 .* randn.()
    res_true = lsim(P_true, u, t)
    d = iddata(res_true)

    nx = 2
    function regularizer(p, _)
        sum(abs2, p.K)
    end

    res = ControlSystemIdentification.structured_pem(d, nx; focus=:prediction, p0, constructor, regularizer, show_trace = false, show_every = 1, iterations=300)
    p_est = res.res.minimizer.p
    @test norm(p_est - p_true) < 1e-6

    # res = ControlSystemIdentification.structured_pem(d, nx; focus=:prediction, p0=0.98p_true, constructor, regularizer, show_trace = true, show_every = 1, iterations=300000, h=6, optimizer=NelderMead())
    # p_est = res.res.minimizer.p
    # @test norm(p_est - p_true)/norm(p_true) < 1e-2 # Severe convergence problems here, but it does eventually converge using NelderMead The test is deactivated since it takes a long time

    res = ControlSystemIdentification.structured_pem(d, nx; focus=:simulation, p0, constructor, regularizer, show_trace = false, show_every = 1, iterations=300)
    p_est = res.res.minimizer.p
    @test norm(p_est - p_true) < 1e-6
    
end