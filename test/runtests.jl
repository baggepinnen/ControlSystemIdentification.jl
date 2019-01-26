using ControlSystemIdentification, ControlSystems, Optim
using Test, Random, LinearAlgebra

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

@testset "ControlSystemIdentification.jl" begin
    @testset "pem" begin
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

        # using BenchmarkTools
        # @btime begin
        # Random.seed!(0)
        sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs2)
        # end
        # 462ms 121 29
        # 296ms
        # 283ms


        @test sysh.C*x0h ≈ sys.C*x0 atol=0.1

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
        @test sysh.C*x0h ≈ sys.C*x0 atol=0.1
        @test Optim.minimum(opt) < 2σy^2*T # A factor of 2 margin

        # Only input noise
        σu = 0.1
        σy = 0.0

        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs2)
        @test sysh.C*x0h ≈ sys.C*x0 atol=0.1
        @test Optim.minimum(opt) < 1 # Should depend on system gramian, but too lazy to figure out


        # Both noises
        σu = 0.2
        σy = 0.2

        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        sysh,x0h,opt = pem(yn,un,nx=3nx, focus=:prediction, metric=abs2, iterations=400)
        @test sysh.C*x0h ≈ sys.C*x0 atol=1
        @test Optim.minimum(opt) < 2σy^2*T # A factor of 2 margin

        # Simulation error minimization
        σu = 0.0
        σy = 0.0

        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        sysh,x0h,opt = pem(yn,un,nx=nx, focus=:simulation, metric=abs2)
        @test sysh.C*x0h ≈ sys.C*x0 atol=0.1
        @test Optim.minimum(opt) < 1

        # L1 error minimization
        σu = 0.0
        σy = 0.0

        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs, regularizer=p->0.1norm(p))
        @test sysh.C*x0h ≈ sys.C*x0 atol=0.1
        @test Optim.minimum(opt) < 1

        yh = ControlSystemIdentification.predict(sysh, yn, u, x0h)
        @test sum(abs2,y-yh) < 0.1

        yh = ControlSystemIdentification.simulate(sysh, u, x0h)
        @test sum(abs2,y-yh) < 0.1
    end

    @testset "arx" begin
        N = 20
        t = 1:N
        u = randn(N)
        G = tf(0.8, [1,-0.9], 1)
        y = lsim(G,u,t)[1][:]

        na,nb = 1,1
        yr,A = getARXregressor(y,u,na,nb)
        @test length(yr) == N-na
        @test size(A) == (N-na, na+nb)

        @test yr == y[na+1:end]
        @test A[:,1] == y[1:end-na]
        @test A[:,2] == u[1:end-1]

        na = 2
        Gh,Σ = arx(1,y,u,na,nb)
        @test Gh ≈ G # Should recover the original transfer function exactly
        ω=exp10.(range(-2, stop=1, length=200))
        # bodeplot(G,ω)
        # bodeconfidence!(Gh, Σ, ω=ω, color=:blue, lab="Est")

        # Test MISO estimation
        u2 = randn(N)
        G2 = [G tf(0.5, [1, -0.9], 1)]
        y2 = lsim(G2,[u u2],t)[1][:]

        nb = [1,1]
        Gh2,Σ = arx(1,y2,[u u2],na,nb)

        @test Gh2 ≈ G2
    end


    @testset "plr" begin

        N = 2000
        t = 1:N
        u = randn(N)
        G = tf(0.8, [1,-0.9], 1)
        y = lsim(G,u,t)[1][:]
        e = randn(N)
        yn = y + e

        na,nb,nc = 2,1,1

        Gls,Σ = arx(1,yn,u,na,nb)
        Gtls,Σ = arx(1,yn,u,na,nb, estimator=tls)
        Gwtls,Σ = arx(1,yn,u,na,nb, estimator=wtls_estimator(y,na,nb))

        Gplr, Gn, ehat = ControlSystemIdentification.plr(1,yn,u,na,nb,nc, initial_order=20)
        @show Gplr, Gn

    end

end
