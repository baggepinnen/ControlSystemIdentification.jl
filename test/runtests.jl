using ControlSystemIdentification, ControlSystems, Optim
using Test, Random, LinearAlgebra

function ⟂(x)
    u,s,v = svd(x)
    u*v
end
function generate_system(nx,ny,nu)
    U,S  = ⟂(randn(nx,nx)), diagm(0=>0.2 .+ 0.78rand(nx))
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
        # 173ms
        @test sysh.C*x0h ≈ sys.C*x0 atol=0.1
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
        @time sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=abs2)
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
        σu = 0.01
        σy = 0.01

        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        @time sysh,x0h,opt = pem(yn,un,nx=nx, focus=:simulation, metric=abs2)
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
        Gplr, Gn = plr(1,yn,u,na,nb,nc, initial_order=20)
        @show Gplr, Gn

    end
    # end
    @testset "frd" begin
        Random.seed!(1)
        ##
        T   = 10000
        sim(sys,u) = lsim(sys, u, 1:T)[1][:]
        σy = 0.1
        sys = tf(1,[1,2*0.1,0.1])
        sysn = tf(σy,[1,2*0.1,0.1])

        u  = randn(T)
        y  = sim(sys, u)
        yn = y + sim(sysn, randn(size(u)))

        # using BenchmarkTools
        # @btime begin
        # Random.seed!(0)
        k = coherence(1,y,u)
        @test all(k.r .> 0.99)
        k = coherence(1,yn,u)
        @test all(k.r .> 0.9)
        G = tfest(1,yn,u)
        # bodeplot([sys,sysn], exp10.(range(-3, stop=log10(pi), length=200)), layout=(3,1), plotphase=false, subplot=[1,3])

        # coherenceplot!(1,yn,u, subplot=2)
        # plot!(G, subplot=1, lab="G Est", alpha=0.4)
        # plot!(N, subplot=3, lab="N Est", alpha=0.4)

    end
end


# 
# Random.seed!(1)
# ##
# T   = 300
# nx  = 3
# nu  = 1
# ny  = 1
# x0  = randn(nx)
# sim(sys,u,x0=x0) = lsim(sys, u', 1:T, x0=x0)[1]'
# sys = generate_system(nx,nu,ny)
# sysn = generate_system(nx,nu,ny)
#
# σy = 0.1
#
# u  = randn(nu,T)
# y  = sim(sys, u, x0)
# yn = y + sim(sysn, σy*randn(size(u)))
#
# uv  = randn(nu,T)
# yv  = sim(sys, uv, x0)
# ynv = yv + sim(sysn, σy*randn(size(uv)))
#
# using GenericLinearAlgebra
# regularizer = function(p)
#     model = ControlSystemIdentification.model_from_params(p,nx,ny,nu)
#     s = model.sys
#     nm = noise_model(s)
#     e = abs.(eigvals(s.A-s.K*s.C))
#     10sum((e.-0.9).* (e .> 0.9)) +
#     10maximum(abs.(freqresp(nm, exp10.(range(-3, stop=3, length=200)))))
# end
#
#
# ##
# fig = plot(layout=4)
# # res = [pem(yn,u,nx=nx+i, focus=:prediction, regularizer=p->10norm(p[end-ny*(nx+i)+1:end])) for i = 0:2]
# res = [pem(yn,u,nx=nx+i, focus=:prediction, iterations=400, regularizer=regularizer) for i = 0:2]
# for i in eachindex(res)
#     (sysh,x0h,opt) = res[i]
#     ControlSystemIdentification.compareplot!(sysh,ynv,uv,x0h; subplot=1, ploty=i==1)
#     ControlSystemIdentification.predplot!(sysh,ynv,uv,x0h; subplot=2, ploty=i==1)
# end
# bodeplot!(ss.(getindex.(res,1)), plotphase=false, subplot=3)
# bodeplot!(noise_model.(getindex.(res,1)), plotphase=false, subplot=4)
# display(fig)
