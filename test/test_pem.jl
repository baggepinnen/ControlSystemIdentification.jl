using ControlSystemIdentification
Random.seed!(1)
T = 1000
nx = 3
nu = 1
ny = 1
x0 = randn(nx)
sim(sys, u, x0 = x0) = lsim(sys, u, 1:T, x0 = x0)[1]
sys = c2d(ControlSystems.DemoSystems.resonant() * tf(1, [0.1, 1]), 1)# generate_system(nx, nu, ny)
sysn = generate_system(nx, nu, ny)

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
# sysh, x0h, opt = pem(d, nx = nx, focus = :prediction, iterations=5000)
sysh, opt = ControlSystemIdentification.newpem(d, nx, optimizer=LBFGS())
# bodeplot([sys,ss(sysh)], exp10.(range(-3, stop=log10(pi), length=150)), legend=false, ylims=(0.01,100))
# end
# 462ms 121 29
# 296ms
# 283ms
# 173ms
# @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test freqresptest(sys, sysh.sys) < 1e-2
# yh = sim(sysh, u, x0h)
@test Optim.minimum(opt) < 1e-4

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
sysh, opt = ControlSystemIdentification.newpem(d, nx, focus = :prediction)
# @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
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
# @time sysh, x0h, opt = pem(d, nx = nx, focus = :prediction)
sysh, opt = ControlSystemIdentification.newpem(d, nx, focus = :prediction)
# @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test Optim.minimum(opt) < 1e-3 # Should depend on system gramian, but too lazy to figure out
@test freqresptest(sys, sysh.sys) < 0.5

# Both noises
σu = 0.2
σy = 0.2

u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
# sysh, x0h, opt = pem(d, nx = 3nx, focus = :prediction, iterations = 400)
sysh, opt = ControlSystemIdentification.newpem(d, 2nx, focus = :prediction, optimizer=NelderMead())
# @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
@test Optim.minimum(opt) < 2σy^2  # A factor of 2 margin
@test freqresptest(sys, sysh) < 1e-1

# Simulation error minimization
σu = 0.01
σy = 0.01

u = randn(nu, T)
un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
y = sim(sys, un, x0)
yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
d = iddata(yn, un, 1)
# @time sysh, x0h, opt = pem(d, nx = nx, focus = :simulation)
@time sysh, opt = ControlSystemIdentification.newpem(d, nx, focus = :simulation)
# @test sysh.C * x0h ≈ sys.C * x0 atol = 0.3
@test Optim.minimum(opt) < 0.01
# @test freqresptest(sys, sysh) < 1e-1