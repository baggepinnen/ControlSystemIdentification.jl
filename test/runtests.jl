using ControlSystemIdentification, ControlSystems, Optim
using Test, Random

function ⟂(x)
    u,s,v = svd(x)
    u*v
end
function generate_system(nx,ny,nu)
    U,S  = ⟂(randn(nx,nx)), diagm(0=>0.2 .+ 0.5rand(nx))
    A    = S*U
    B   = randn(nx,nu)
    C   = randn(ny,nx)
    sys = ss(A,B,C,0,1)
end

# @testset "ControlSystemIdentification.jl" begin
Random.seed!(1)
T   = 1000
nx  = 3
nu  = 1
ny  = 1
x0  = randn(nx)
sim(sys,u,x0=x0) = lsim(sys, u', 1:T, x0=x0)[1]'
sys = generate_system(nx,nu,ny)
sysn = generate_system(nx,nu,ny)

σu = 0
σy = 0

u  = randn(nu,T)
un = u + sim(sysn, σu*randn(size(u)),0*x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy*randn(size(u)),0*x0)

sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs2)

@test ControlSystems.get_C(sysh)*x0h ≈ sys.C*x0 atol=0.1

G = tf(sys)
H = tf(convert(StateSpace,sysh))

#
yh = sim(convert(StateSpace,sysh), u, x0h)
@test Optim.minimum(opt) < 1 # Should reach 0


# Test with some noise

# Only measurement noise
σu = 0.0
σy = 0.1

u  = randn(nu,T)
un = u + sim(sysn, σu*randn(size(u)),0*x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy*randn(size(u)),0*x0)
sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs2)
@test ControlSystems.get_C(sysh)*x0h ≈ sys.C*x0 atol=0.1
@test Optim.minimum(opt) < 2σy^2*T # A factor of 2 margin

# Only input noise
σu = 0.1
σy = 0.0

u  = randn(nu,T)
un = u + sim(sysn, σu*randn(size(u)),0*x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy*randn(size(u)),0*x0)
sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs2)
@test ControlSystems.get_C(sysh)*x0h ≈ sys.C*x0 atol=0.1
@test Optim.minimum(opt) < 1 # Should depend on system gramian, but too lazy to figure out


# Both noises
σu = 0.1
σy = 0.1

u  = randn(nu,T)
un = u + sim(sysn, σu*randn(size(u)),0*x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy*randn(size(u)),0*x0)
sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs2)
@test ControlSystems.get_C(sysh)*x0h ≈ sys.C*x0 atol=0.1
@test Optim.minimum(opt) < 2σy^2*T # A factor of 2 margin

# Simulation error minimization
σu = 0.0
σy = 0.0

u  = randn(nu,T)
un = u + sim(sysn, σu*randn(size(u)),0*x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy*randn(size(u)),0*x0)
sysh,x0h,opt = pem(yn,un,nx=nx, focus=:simulation, metric=abs2)
@test ControlSystems.get_C(sysh)*x0h ≈ sys.C*x0 atol=0.1
@test Optim.minimum(opt) < 1 # A factor of 2 margin

# L1 error minimization
σu = 0.0
σy = 0.0

u  = randn(nu,T)
un = u + sim(sysn, σu*randn(size(u)),0*x0)
y  = sim(sys, un, x0)
yn = y + sim(sysn, σy*randn(size(u)),0*x0)
sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs)
@test ControlSystems.get_C(sysh)*x0h ≈ sys.C*x0 atol=0.1
@test Optim.minimum(opt) < 1 # A factor of 2 margin

yh = ControlSystemIdentification.predict(sysh, yn, u, x0h)

# end
