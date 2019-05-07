using ControlSystemIdentification, ControlSystems, Optim, Plots, DSP
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
        sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction)
        # bodeplot([sys,ss(sysh)], exp10.(range(-3, stop=log10(pi), length=150)), legend=false, ylims=(0.01,100))
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
        sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction)
        @test sysh.C*x0h ≈ sys.C*x0 atol=0.1
        @test Optim.minimum(opt) < 2σy^2*T # A factor of 2 margin

        # Only input noise
        σu = 0.1
        σy = 0.0
        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        @time sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction)
        @test sysh.C*x0h ≈ sys.C*x0 atol=0.1
        @test Optim.minimum(opt) < 1 # Should depend on system gramian, but too lazy to figure out


        # Both noises
        σu = 0.2
        σy = 0.2

        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        sysh,x0h,opt = pem(yn,un,nx=3nx, focus=:prediction, iterations=400)
        @test sysh.C*x0h ≈ sys.C*x0 atol=1
        @test Optim.minimum(opt) < 2σy^2*T # A factor of 2 margin

        # Simulation error minimization
        σu = 0.01
        σy = 0.01

        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        @time sysh,x0h,opt = pem(yn,un,nx=nx, focus=:simulation)
        @test sysh.C*x0h ≈ sys.C*x0 atol=0.3
        @test Optim.minimum(opt) < 1

        # L1 error minimization
        σu = 0.0
        σy = 0.0

        u  = randn(nu,T)
        un = u + sim(sysn, σu*randn(size(u)),0*x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy*randn(size(u)),0*x0)
        sysh,x0h,opt = pem(yn,un,nx=nx, focus=:prediction, metric=e->sum(abs,e), regularizer=p->(0.1/T)*norm(p))
        # 409ms
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

        pars = ControlSystemIdentification.params(G)
        @test pars == ([0.9,0.8],[0.9],[0.8])
        @test ControlSystemIdentification.params2poly(pars[1],1,1) == ([1,-0.9], [0.8])

        na,nb = 1,1
        yr,A = getARXregressor(y,u,na,nb)
        @test length(yr) == N-na
        @test size(A) == (N-na, na+nb)

        @test yr == y[na+1:end]
        @test A[:,1] == y[1:end-na]
        @test A[:,2] == u[1:end-1]

        na = 1
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

    @testset "ar" begin
        N = 10000
        t = 1:N
        y = zeros(N)
        y[1] = randn()
        for i = 2:N
            y[i] = 0.9y[i-1]
        end
        G = tf(1, [1,-0.9], 1)

        na = 1
        yr,A = ControlSystemIdentification.getARregressor(y,na)
        @test length(yr) == N-na
        @test size(A) == (N-na, na)

        @test yr == y[na+1:end]
        @test A[:,1] == y[1:end-na]

        Gh,Σ = ar(1,y,na)
        @test Gh ≈ G # We should be able to recover this transfer function

        N = 10000
        t = 1:N
        y = zeros(N)
        y[1] = 5randn()
        for i = 2:N
            y[i] = 0.9y[i-1] + 0.01randn()
        end
        Gh,Σ = ar(1,y,na)
        @test Gh ≈ G atol=0.02 # We should be able to recover this transfer function
        yh = predict(Gh,y)
        @test rms(y[2:end]-yh) < 0.0102

    end


    @testset "plr" begin

        N = 2000
        t = 1:N
        u = randn(N)
        G = tf(0.8, [1,-0.9], 1)
        y = lsim(G,u,t)[1][:]
        e = randn(N)
        yn = y + e

        na,nb,nc = 1,1,1
        find_na(y,6)
        find_nanb(y,u,6,6)
        Gls,Σ = arx(1,yn,u,na,nb)
        Gtls,Σ = arx(1,yn,u,na,nb, estimator=tls)
        Gwtls,Σ = arx(1,yn,u,na,nb, estimator=wtls_estimator(y,na,nb))
        Gplr, Gn = plr(1,yn,u,na,nb,nc, initial_order=20)
        bodeconfidence(Gplr, Σ, exp10.(range(-3, stop=log10(pi), length=150)))
        @show Gplr, Gn

    end
    # end
    @testset "frd" begin
        Random.seed!(1)
        ##
        T   = 100000
        h   = 1
        t = range(0,step=h, length=T)
        sim(sys,u) = lsim(sys, u, t)[1][:]
        σy = 0.5
        sys = tf(1,[1,2*0.1,0.1])
        ωn = sqrt(0.3)
        sysn = tf(σy*ωn,[1,2*0.1*ωn,ωn^2])

        u  = randn(T)
        y  = sim(sys, u)
        yn = y + sim(sysn, randn(size(u)))

        # using BenchmarkTools
        # @btime begin
        # Random.seed!(0)
        k = coherence(h,y,u)
        @test all(k.r .> 0.99)
        k = coherence(h,yn,u)
        @test all(k.r[1:10] .> 0.9)
        @test k.r[end] .> 0.8
        @test k.r[findfirst(k.w .> ωn)] < 0.6
        G,N = tfest(h,yn,u, 0.02)
        noisemodel = innovation_form(ss(sys), syse=ss(sysn))
        noisemodel.D .*= 0
        bodeplot([sys,sysn], exp10.(range(-3, stop=log10(pi), length=200)), layout=(1,3), plotphase=false, subplot=[1,2], size=(3*800, 600), linecolor=:blue)#, ylims=(0.1,300))

        coherenceplot!(h,yn,u, subplot=3)
        plot!(G, subplot=1, lab="G Est", alpha=0.3, title="Process model")
        plot!(√N, subplot=2, lab="N Est", alpha=0.3, title="Noise model")

    end


    @testset "plots and difficult" begin

        Random.seed!(1)
        ##
        T   = 200
        nx  = 2
        nu  = 1
        ny  = 1
        x0  = randn(nx)
        σy = 0.5
        sim(sys,u) = lsim(sys, u', 1:T)[1]'
        sys = tf(1,[1,2*0.1,0.1])
        sysn = tf(σy,[1,2*0.1,0.3])

        u  = randn(nu,T)
        un = u + 0.1randn(size(u))
        y  = sim(sys, u)
        yn = y + sim(sysn, σy*randn(size(u)))

        uv  = randn(nu,T)
        yv  = sim(sys, uv)
        ynv = yv + sim(sysn, σy*randn(size(uv)))
        ##

        res = [pem(yn,un,nx=nx, iterations=50, difficult=true, focus=:prediction) for nx = [1,3,4]]

        ω = exp10.(range(-2, stop=log10(pi), length=150))
        fig = plot(layout=4, size=(1000,600))
        for i in eachindex(res)
            (sysh,x0h,opt) = res[i]
            ControlSystemIdentification.simplot!(sysh,ynv,uv,x0h; subplot=1, ploty=i==1)
            ControlSystemIdentification.predplot!(sysh,ynv,uv,x0h; subplot=2, ploty=i==1)
        end
        bodeplot!(ss.(getindex.(res,1)), ω, plotphase=false, subplot=3, title="Process", linewidth=2*[4 3 2 1])
        bodeplot!(ControlSystems.innovation_form.(getindex.(res,1)), ω, plotphase=false, subplot=4, linewidth=2*[4 3 2 1])
        bodeplot!(sys, ω, plotphase=false, subplot=3, lab="True", linecolor=:blue, l=:dash, legend = :bottomleft, title="System model")
        bodeplot!(ControlSystems.innovation_form(ss(sys),syse=ss(sysn), R2=σy^2*I), ω, plotphase=false, subplot=4, lab="True", linecolor=:blue, l=:dash, ylims=(0.1, 100), legend = :bottomleft, title="Noise model")
        display(fig)

    end


    @testset "impulseest" begin
        T = 200
        h = 0.1
        t = h:h:T
        sim(sys,u) = lsim(sys, u, t)[1][:]
        sys = c2d(tf(1,[1,2*0.1,0.1]),h)

        u  = randn(length(t))
        y  = sim(sys, u) + 0.1randn(length(t))

        impulseestplot(h,y,u,Int(50/h), 0)
        impulseplot!(sys,50, l=(:dash,:blue))
    end

end
