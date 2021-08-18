using ControlSystemIdentification, ControlSystems, Optim, Plots, DSP, TotalLeastSquares
using Test, Random, LinearAlgebra, Statistics

using ControlSystemIdentification: time1, time2
using MonteCarloMeasurements

function ⟂(x)
    u, s, v = svd(x)
    u * v
end
function generate_system(nx, ny, nu)
    U, S = ⟂(randn(nx, nx)), diagm(0 => 0.2 .+ 0.78rand(nx))
    A    = S * U
    B    = randn(nx, nu)
    C    = randn(ny, nx)
    sys  = ss(A, B, C, 0, 1)
end


wtest = exp10.(LinRange(-3, log10(pi), 30))
freqresptest(G, model) =
    maximum(abs, log10.(abs2.(freqresp(model, wtest))) - log10.(abs2.(freqresp(G, wtest))))

freqresptest(G, model, tol) = freqresptest(G, model) < tol

@testset "ControlSystemIdentification.jl" begin

    @testset "basis functions" begin
        @info "Testing basis functions"
        include("test_basis_functions.jl")
    end

    @testset "FRD arma fit" begin
        @info "Testing FRD arma fit"
        include("test_frq_tf.jl")
    end


    @testset "Frequency weights" begin
        @info "Testing Frequency weights"
        include("test_frequency_weights.jl")
    end


    @testset "Utils" begin
        @info "Testing Utils"
        v = zeros(2)
        M = zeros(2, 4)
        vv = fill(v, 4)
        @test time1(v) == v
        @test time1(M) == M'
        @test time1(vv) == M'
        @test time2(v) == v'
        @test time2(M) == M
        @test time2(vv) == M

        v = zeros(2)
        M = zeros(1, 2)
        vv = [[0.0], [0.0]]
        for x in (v, M, vv), t in (M, vv)
            # @show typeof(t), typeof(x)
            @test oftype(t, t) == t
            @test typeof(oftype(t, x)) == typeof(t)
        end

        y = randn(1, 3)
        @test ControlSystemIdentification.modelfit(y, y) == [100]

    end

    @testset "iddata" begin
        @info "Testing iddata"
        @testset "vectors" begin
            T = 100
            y = randn(1, T)
            @show d = iddata(y)
            @test d isa ControlSystemIdentification.OutputData
            @test length(d) == T
            @test output(d) == y
            @test !hasinput(d)
            @test ControlSystemIdentification.time2(y) == y
            @test sampletime(d) == 1
            @test d[1:10] isa typeof(d)
            @test length(d[1:10]) == 10
            @test length(timevec(d)) == length(d)

            @test_nowarn plot(d)

            u = randn(1, T)
            @show d = iddata(y, u)
            @test d isa ControlSystemIdentification.InputOutputData
            @test length(d) == T
            @test output(d) == y
            @test hasinput(d)
            @test input(d) == u
            @test ControlSystemIdentification.time2(y) == y
            @test sampletime(d) == 1
            @test d[1, 1] == d
            @test d[1:10] isa typeof(d)

            @test length([d d]) == 2length(d)

            @test oftype(Matrix, output(d)) == y

            yr, A = getARXregressor(d, 2, 2)
            @test size(A, 2) == 4

            @test_nowarn plot(d)
        end

        @testset "matrices" begin
            T = 10
            ny, nu = 2, 3
            y = randn(ny, T)
            @show d = iddata(y)
            @test d isa ControlSystemIdentification.OutputData
            @test length(d) == T
            @test output(d) == y
            @test !hasinput(d)
            @test ControlSystemIdentification.time1(y) == y'
            @test sampletime(d) == 1

            @test_nowarn plot(d)

            u = randn(nu, T)
            @show d = iddata(y, u)
            @test d isa ControlSystemIdentification.InputOutputData
            @test length(d) == T
            @test output(d) == y
            @test hasinput(d)
            @test input(d) == u
            @test sampletime(d) == 1

            @test_nowarn plot(d)

            u = randn(T, nu)
            @show d = iddata(y, u, 2)
            @test d isa ControlSystemIdentification.InputOutputData
            @test length(d) == T
            @test output(d) == y
            @test hasinput(d)
            @test input(d) == u'
            @test ControlSystemIdentification.time1(input(d)) == u
            @test sampletime(d) == 2

            @test_nowarn plot(d)

        end

        @testset "vectors of vectors" begin
            T = 100
            ny, nu = 2, 3
            y = [randn(ny) for _ = 1:T]
            @show d = iddata(y)
            @test d isa ControlSystemIdentification.OutputData
            @test length(d) == T
            @test output(d) == y
            @test !hasinput(d)
            @test ControlSystemIdentification.time1(y) == reduce(hcat, y)'
            @test sampletime(d) == 1

            @test_nowarn plot(d)

            u = [randn(nu) for _ = 1:T]
            @show d = iddata(y, u)
            @test d isa ControlSystemIdentification.InputOutputData
            @test length(d) == T
            @test output(d) == y
            @test hasinput(d)
            @test input(d) == u
            @test sampletime(d) == 1

            u = randn(T, nu)
            @show d = iddata(y, u, 2)
            @test d isa ControlSystemIdentification.InputOutputData
            @test length(d) == T
            @test output(d) == y
            @test hasinput(d)
            @test input(d) == u'
            @test ControlSystemIdentification.time1(y) == reduce(hcat, y)'
            @test ControlSystemIdentification.time1(input(d)) == u
            @test sampletime(d) == 2

            @test_nowarn plot(d)

        end



    end


    @testset "n4sid" begin
        @info "Testing n4sid"


        N = 200
        r = 2
        m = 2
        l = 2
        for r = 1:5, m = 1:2, l = 1:2
            Random.seed!(0)
            @show r, m, l
            A = Matrix{Float64}(I(r))
            A[1, 1] = 1.01
            G = ss(A, randn(r, m), randn(l, r), 0 * randn(l, m), 1)
            u = randn(N, m)
            x0 = randn(r)
            y, t, x = lsim(G, u, 1:N, x0 = x0)
            @test sum(!isfinite, y) == 0
            yn = y + 0.1randn(size(y))
            d = iddata(yn, u, 1)
            res = n4sid(d, r, γ = 0.99)
            @test maximum(abs, pole(res.sys)) <= 1.00001 * 0.99

            ys = simulate(res, d, res.x[:, 1])
            # @show mean(abs2,y-ys') / mean(abs2,y)
            ys = simulate(res, d, res.x[:, 1], stochastic = true)
            # @show mean(abs2,y-ys') / mean(abs2,y)

            yp = predict(res, d, res.x[:, 1])
            # @show mean(abs2,y-yp') / mean(abs2,y)
            @test mean(abs2, y - yp') / mean(abs2, y) < 0.01


            res = n4sid(d, r, i = r + 1)
            freqresptest(G, res.sys) < 0.2 * m * l
            w = exp10.(LinRange(-5, log10(pi), 600))
            bodeplot(G, w)
            bodeplot!(res.sys, w)


            G = ss(0.2randn(r, r), randn(r, m), randn(l, r), 0 * randn(l, m), 1)
            u = randn(N, m)

            y, t, x = lsim(G, u, 1:N, x0 = randn(r))
            @assert sum(!isfinite, y) == 0
            ϵ = 0.01
            yn = y + ϵ * randn(size(y))
            d = iddata(yn, u)

            res = n4sid(d, r)
            @test res.sys.nx == r
            @test sqrt.(diag(res.R)) ≈ ϵ * ones(l) rtol = 0.5
            @test norm(res.S) < ϵ
            @test freqresptest(G, res.sys) < 0.2 * m * l

            if m == l # == 1
                iy, it, ix = impulse(G, 20)
                H = okid(d, r, 20)
                @test norm(iy - permutedims(H, (3, 1, 2))) < 0.05
                # plot([iy vec(H)])

                sys = era(d, r)
                @test sys.nx == r
                @test freqresptest(G, sys) < 0.2 * m * l
            end

            res = ControlSystemIdentification.n4sid(d)
            @test res.sys.nx <= r # test that auto rank selection don't choose too high rank when noise is low
            kf = KalmanFilter(res)
            @test kf isa KalmanFilter
            @test kf.A == res.A
            @test kf.B == res.B
            @test kf.C == res.C

        end

        m = l = r = 1
        for a in LinRange(-0.99, 0.99, 10), b in LinRange(-5, 5, 10)
            G = tf(b, [1, -a], 1)
            u = randn(N, m)
            y, t, x = lsim(G, u, 1:N, x0 = randn(r))
            yn = y + 0.01randn(size(y))
            d = iddata(yn, u)
            res = n4sid(d, r)
            @test res.sys.A[1] ≈ a atol = 0.01
            @test numvec(tf(res.sys))[1][2] ≈ b atol = 0.01
            @test abs(numvec(tf(res.sys))[1][1]) < 1e-2
            @test freqresptest(G, res.sys) < 0.01
        end


    end


    @testset "pem" begin
        Random.seed!(1)
        T = 1000
        nx = 3
        nu = 1
        ny = 1
        x0 = randn(nx)
        sim(sys, u, x0 = x0) = lsim(sys, u', 1:T, x0 = x0)[1]'
        sys = generate_system(nx, nu, ny)
        sysn = generate_system(nx, nu, ny)

        σu = 0
        σy = 0

        u  = randn(nu, T)
        un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
        y  = sim(sys, un, x0)
        yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
        d  = iddata(yn, un, 1)

        # using BenchmarkTools
        # @btime begin
        # Random.seed!(0)
        sysh, x0h, opt = pem(d, nx = nx, focus = :prediction, iterations=5000)
        # bodeplot([sys,ss(sysh)], exp10.(range(-3, stop=log10(pi), length=150)), legend=false, ylims=(0.01,100))
        # end
        # 462ms 121 29
        # 296ms
        # 283ms
        # 173ms
        @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
        @test freqresptest(sys, StateSpace(sysh)) < 1e-7
        yh = sim(convert(StateSpace, sysh), u, x0h)
        @test Optim.minimum(opt) < 1 # Should reach 0

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
        @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
        @test Optim.minimum(opt) < 2σy^2 * T # A factor of 2 margin

        # Only input noise
        σu = 0.1
        σy = 0.0
        u = randn(nu, T)
        un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
        y = sim(sys, un, x0)
        yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
        d = iddata(yn, un, 1)
        @time sysh, x0h, opt = pem(d, nx = nx, focus = :prediction)
        @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
        @test Optim.minimum(opt) < 1 # Should depend on system gramian, but too lazy to figure out


        # Both noises
        σu = 0.2
        σy = 0.2

        u = randn(nu, T)
        un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
        y = sim(sys, un, x0)
        yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
        d = iddata(yn, un, 1)
        sysh, x0h, opt = pem(d, nx = 3nx, focus = :prediction, iterations = 400)
        @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
        @test Optim.minimum(opt) < 2σy^2  # A factor of 2 margin

        # Simulation error minimization
        σu = 0.01
        σy = 0.01

        u = randn(nu, T)
        un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
        y = sim(sys, un, x0)
        yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
        d = iddata(yn, un, 1)
        @time sysh, x0h, opt = pem(d, nx = nx, focus = :simulation)
        @test sysh.C * x0h ≈ sys.C * x0 atol = 0.3
        @test Optim.minimum(opt) < 0.01

        # L1 error minimization
        σu = 0.01
        σy = 0.01

        u = randn(nu, T)
        un = u + sim(sysn, σu * randn(size(u)), 0 * x0)
        y = sim(sys, un, x0)
        yn = y + sim(sysn, σy * randn(size(u)), 0 * x0)
        d = iddata(yn, un, 1)
        sysh, x0h, opt = pem(
            d,
            nx = nx,
            focus = :prediction,
            metric = e -> sum(abs, e),
            regularizer = p -> (0.1 / T) * norm(p),
        )
        # 409ms
        @test sysh.C * x0h ≈ sys.C * x0 atol = 0.1
        @test Optim.minimum(opt) < 0.01

        yh = ControlSystemIdentification.predict(sysh, yn, u, x0h)
        @test mean(abs2, y - yh) < 0.01

        yh = ControlSystemIdentification.simulate(sysh, u, x0h)
        @test mean(abs2, y - yh) < 0.01

        yh = ControlSystemIdentification.predict(sysh, iddata(yn, u), x0h)
        @test mean(abs2, y - yh) < 0.01
    end

    """
    Compare a tf to a tf based on particles .. maybe replaye by correct dispatched isapprox
    """
    function compareTFs(tf, tfStochastic, atol = 10e-5)
        aok = bok = false
        try
            bok = isapprox(numvec(tf), mean.(numvec(tfStochastic)), atol = atol)
            aok = isapprox(denvec(tf), mean.(denvec(tfStochastic)), atol = atol)
        catch
            return false
        end  
            return aok && bok
    end

    @testset "arx" begin
        @info "Testing arx"

        # Test params2poly2
        y = randn(10)
        u = randn(10)
        for na in 1:3, nb in 1:3, inputdelay in 0:2 
            Y, A = getARXregressor(y, u, na, nb, inputdelay = inputdelay)
            w = \(A, Y)
            a, b = ControlSystemIdentification.params2poly2(w, na, nb)
            @test length(a) == length(b[1])
            @test length(w) == na + nb
        end
        
        unsafe_comparisons(true)

        N = 20
        t = 1:N
        u = randn(N)
        G = tf(0.8, [1, -0.9], 1)
        y = lsim(G, u, t)[1][:]

        pars = ControlSystemIdentification.params(G)
        @test pars == ([0.9, 0.8], [0.9], [[0.8]], [1])
        @test ControlSystemIdentification.params2poly(pars[1], 1, 1) == ([1, -0.9], [[0.8]])

        na, nb = 1, 1
        yr, A = getARXregressor(y, u, na, nb)
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

        # Test MISO estimation
        u2 = randn(N)
        G2 = [G tf(0.5, [1, -0.9], 1)]
        y2 = lsim(G2, [u u2], t)[1][:]

        nb = [1, 1]
        d = iddata(y2, [u u2], 1)
        Gh2 = arx(d, na, nb)
        @test Gh2 ≈ G2
        Gh2s = arx(d, na, nb, stochastic = true)
        @test compareTFs(G2, Gh2s)
        
        # SISO
        ## Test na < nb
        na, nb = 1, 2
        G1 =  tf([1, -2], [1, -0.5, 0], 1)
        u = randn(N)
        y = lsim(G1, u, t)[1][:]
        d = iddata(y, u, 1)
        Gest = arx(d, na, nb)
        @test G1 ≈ Gest
        
        Gests = arx(d, na, nb, stochastic = true)
        @test compareTFs(G1, Gests)

        ## Test na > nb
        na, nb = 2, 1
        G1  = tf([0.8, 0], [1,0.5,0.25], 1)
        u = randn(N)
        y = lsim(G1, u, t)[1][:]
        d = iddata(y, u, 1)
        Gest = arx(d, na, nb)
        @test G1 ≈ minreal(Gest)

        # MISO nb1 != nb2
        na, nb = 1, [2, 3]
        G1 =  tf([1, -2], [1, -0.5, 0], 1)
        G2 =  tf([1, -2, 3], [1, -0.5, 0, 0], 1)
        G = [G1 G2]
        u1 = randn(N)
        u2 = randn(N)
        u = [u1 u2]
        y = lsim(G, u, t)[1][:]
        d = iddata(y, u, 1)
        Gest = arx(d, na, nb)
        @test Gest ≈ G

        Gests = arx(d, na, nb, stochastic = true)
        @test compareTFs(G, Gests)
        
        
        # Test na = 0
        na, nb = 0, 1
        G1 =  tf([1], [1, 0], 1)
        u = randn(N)
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
        u = randn(N)
        y = lsim(G1, u, t)[1][:]
        d = iddata(y, u, 1)
        Gest = arx(d, na, nb, inputdelay = inputdelay)
        @test Gest ≈ G1
        Gests = arx(d, na, nb, inputdelay = inputdelay, stochastic = true)
        @test compareTFs(G1, Gests)
        

        ## Test na > nb
        na, nb, inputdelay = 2, 1, 2
        G1   = tf(0.8, [1,0.5,0.25], 1)
        u = randn(N)
        y = lsim(G1, u, t)[1][:]
        d = iddata(y, u, 1)
        Gest = arx(d, na, nb, inputdelay = inputdelay)
        @test G1 ≈ minreal(Gest)


        ## MISO
        na, nb, inputdelay = 1, [1, 1], [2, 3]
        G1 =  tf([0,1], [1, -0.5,0], 1)
        G2 =  tf([0,0,1], [1, -0.5,0,0], 1)
        G = [G1 G2]
        u1 = randn(N)
        u2 = randn(N)
        u = [u1 u2]
        y = lsim(G, u, 1:N)[1][:]
        d = iddata(y, u, 1)
        Gest = arx(d, na, nb, inputdelay = inputdelay)
        @test Gest ≈ G

        Gests = arx(d, na, nb, inputdelay = inputdelay, stochastic =  true)
        @test compareTFs(G, Gests)

        # direct input
        G1 =  tf([0.3, 1], [1, -0.5], 1)
        u = randn(N)
        y = lsim(G1, u, t)[1][:]
        d = iddata(y, u, 1)
        na, nb, inputdelay = 1,2,0
        Gest = arx(d, na, nb, inputdelay = inputdelay)
        @test Gest ≈ G1
        
        Gests = arx(d, na, nb, inputdelay = inputdelay, stochastic = true)
        @test compareTFs(G1, Gests)
        
        ## with inputdelay
        G1 =  tf([0.3,0, 1], [1, -0.5, 0], 1)
        u = randn(N)
        y = lsim(G1, u, t)[1][:]
        d = iddata(y, u, 1)
        na, nb, inputdelay = 1,3,0
        Gest = arx(d, na, nb, inputdelay = inputdelay)
        @test Gest ≈ G1

        Gests = arx(d, na, nb, inputdelay = inputdelay, stochastic = true)
        @test compareTFs(G1, Gests)

    end

    @testset "ar" begin
        N = 10000
        t = 1:N
        y = zeros(N)
        y[1] = -0.2
        u = copy(y)
        for i = 2:N
            y[i] = 0.9y[i-1]
        end
        # G = tf(1, [1, -0.9], 1)
        G = tf([1, 0], [1, -0.9], 1)
        y2 = lsim(G, u, t)[1][:]
        @test all(y .== y2)

        na = 1
        yr, A = getARregressor(y, na)
        @test length(yr) == N - na
        @test size(A) == (N - na, na)

        @test yr == y[na+1:end]
        @test A[:, 1] == y[1:end-na]

        d = iddata(y, 1)
        Gh = ar(d, na)
        @test Gh ≈ G # We should be able to recover this transfer function
        uh = lsim(1/Gh, y, t)[1][:]
        @test u ≈ uh atol = eps()
        
        N = 10000
        t = 1:N
        y = zeros(N)
        y[1] = 5randn()
        for i = 2:N
            y[i] = 0.9y[i-1] + 0.01randn()
        end
        d = iddata(y, 1)
        Gh = ar(d, na)
        @test Gh ≈ G atol = 0.02 # We should be able to recover this transfer function
        @test freqresptest(G, Gh, 0.05)
        yh = predict(Gh, y)
        @test rms(y[2:end] - yh) < 0.0102

        Gh2 = ar(d, na, stochastic = true)
        @test denvec(Gh2)[1][end] ≈ denvec(Gh)[1][end]
    end

    @testset "arxar" begin
        @info "Testing arxar"
        # Test examples are taken from Söderstöms paper and compared against it (this is also where the (high) tolerances come from)
        Random.seed!(0)
        N = 500 # Number of samples used for simulation by Söderström
        time = 1:N
        sim(G, u) = lsim(G, u, time)[1][:]
        
        #### S1 ####
        A = tf([1, -0.8], [1, 0], 1)
        B = tf([0, 1], [1, 0], 1)
        G = minreal(B / A)
        D = tf([1, 0.7], [1, 0], 1)
        H = minreal(1 / (D * A))
        
        u = rand(Normal(0, 1), N)
        e = rand(Normal(0, 1), N)
        y = sim(G, u)
        v = sim(H, e)
        yv = y.+ v
        d = iddata(yv, u, 1)
        ###########
        na, nb , nd = 1, 1, 1
        Gest, Hest, res = arxar(d, na, nb, nd, iterations = 10, verbose = true, δmin = 1e-3)
        @test isapprox(Gest, G, atol = 1e-1)
        @test isapprox(Hest, 1/D, atol = 1e-1)
        @test var(res .- e[2:end]) < 1e-1
        
        #### S12 ####
        A = tf([1, -0.7], [1, 0], 1)
        B = tf([0, 1], [1, 0], 1)
        G = minreal(B / A)
        D = tf([1, 0.9], [1, 0], 1)
        H = minreal(1 / (D * A))
        
        u = rand(Normal(0, 1), N)
        e = rand(Normal(0, sqrt(1.2)), N)
        y = sim(G, u)
        v = sim(H, e)
        yv = y.+ v
        d = iddata(yv, u, 1)
        ############
        na, nb , nd = 1, 1, 1
        Gest, Hest, res = arxar(d, na, nb, nd, iterations = 10, verbose = true, δmin = 1e-3)
        @test isapprox(Gest, G, atol = 2e-1)
        @test isapprox(Hest, 1/D, atol = 1e-1)
        
        #### S2 #### structure of Ay = Bu + Ce
        A = tf([1, -0.8], [1, 0], 1)
        B = tf([0, 1], [1, 0], 1)
        G = minreal(B / A)
        D = tf([1, 0.7], [1, 0], 1)
        H = minreal((D / A))
        
        u = rand(Normal(0, 1), N)
        e = rand(Normal(0, 0.1), N)
        y = sim(G, u)
        v = sim(H, e)
        yv = y .+ v
        d = iddata(yv, u, 1)
        ############
        na, nb , nd = 1, 1, 1
        Gest, Hest, res = arxar(d, na, nb, nd, iterations = 10, verbose = true, δmin = 1e-3)
        @test isapprox(Gest, G, atol = 1e-1)
        @test isapprox(Hest, 1/tf([1, -0.49], [1, 0], 1), atol = 1e-1)
        
        #### S10 #### prior knowledge neccessary for identification
        N = 2000
        A = tf([1, -0.5], [1, 0], 1)
        B = tf([0, 1], [1, 0], 1)
        G = minreal(B / A)
        D = tf([1, 0.5], [1,0], 1)
        H = minreal(1 / (D * A))
        
        u = rand(Normal(0, 1), N)
        e = rand(Normal(0, 5), N)
        y = lsim(G, u, 1:N)[1][:]
        v = lsim(H, e, 1:N)[1][:]
        yv = y.+ v
        d = iddata(yv, u, 1)
        ############
        na, nb , nd = 1, 1, 1
        Gest, Hest, res = arxar(d, na, nb, nd, H = 1/D, iterations = 10, verbose = true, δmin = 1e-3)
        @test isapprox(Gest, G, atol = 4e-1)
        @test isapprox(Hest, 1/D, atol = 1e-1)
        @test freqresptest(G, Gest) < 0.5
        
        # MISO
        N = 500
        A = tf([1, -0.8], [1, 0], 1)
        B1 = tf([0, 1], [1, 0], 1)
        B2 = tf([0, -1], [1, 0], 1)
        G1 = minreal(B1 / A)
        G2 = minreal(B2 / A)
        G = [G1 G2]
        D = tf([1, 0.7], [1, 0], 1)
        H = minreal(1 / (D * A))
        
        u1 = rand(Normal(0, 1), N)
        u2 = rand(Normal(0, 1), N)
        u = [u1 u2]
        e = rand(Normal(0, 1), N)
        y = sim(G, u)
        v = sim(H, e)
        yv = y.+ v
        d = iddata(yv, u', 1)
        ###########
        na, nb , nd = 1, [1, 1], 1
        Gest, Hest, res = arxar(d, na, nb, nd, iterations = 10, verbose = true, δmin = 1e-3)
        @test isapprox(Gest, G, atol = 3e-1)
        @test isapprox(Hest, 1/D, atol = 3e-1)
    
        # inputdelay 
        A = tf([1, -0.8], [1, 0], 1)
        B = tf([0, 1], [1, 0, 0], 1)
        G = minreal(B / A)
        D = tf([1, 0.7], [1, 0], 1)
        H = minreal(1 / (D * A))
        
        u = rand(Normal(0, 1), N)
        e = rand(Normal(0, 1), N)
        y = sim(G, u)
        v = sim(H, e)
        yv = y.+ v
        d = iddata(yv, u, 1)
        ###########
        na, nb , nd, inputdelay = 1, 1, 1, 2
        Gest, Hest, res = arxar(d, na, nb, nd, inputdelay = inputdelay, iterations = 10, verbose = true, δmin = 1e-3)
        @test isapprox(Gest, G, atol = 1e-1)
        @test isapprox(Hest, 1/D, atol = 1e-1)
        @test var(res .- e[3:end]) < 1e-1
    end
    
    @testset "plr" begin

        N = 2000
        t = 1:N
        u = randn(N)
        G = tf(0.8, [1, -0.9], 1)
        y = lsim(G, u, t)[1][:]
        e = randn(N)
        yn = y + e

        na, nb, nc = 1, 1, 1
        d = iddata(yn, u, 1)
        find_na(y, 6)
        find_nanb(d, 6, 6)
        Gls = arx(d, na, nb)
        Gtls = arx(d, na, nb, estimator = tls)
        Gwtls = arx(d, na, nb, estimator = wtls_estimator(y, na, nb))
        Gplr, Gn = ControlSystemIdentification.plr(d, na, nb, nc, initial_order = 10)

        # @show Gplr, Gn

        @test freqresptest(G, Gls) < 1.5
        @test freqresptest(G, Gtls) < 1
        @test freqresptest(G, Gwtls) < 0.1
        @test freqresptest(G, Gplr) < 0.1

        d = iddata(y, u, 1)
        Gls = arx(d, na, nb)
        Gtls = arx(d, na, nb, estimator = tls)
        Gwtls = arx(d, na, nb, estimator = wtls_estimator(y, na, nb))
        Gplr, Gn = ControlSystemIdentification.plr(d, na, nb, nc, initial_order = 10)

        @test freqresptest(G, Gls) < sqrt(eps())
        @test freqresptest(G, Gtls) < sqrt(eps())
        @test freqresptest(G, Gwtls) < sqrt(eps())
        @test freqresptest(G, Gplr) < sqrt(eps())

    end

    @testset "arma" begin
        @info "Testing arma"

        N  = 2000     # Number of time steps
        t  = 1:N
        Δt = 1        # Sample time
        u  = randn(N) # A random control input
        a  = 0.9

        G = tf([1, 0.1], [1, -a, 0], 1)
        y, t, x = lsim(G, u, 1:N) .|> vec

        na, nc = 2, 2   # Number of polynomial coefficients
        e = 0.0001randn(N) #+ 20randn(N) .* (rand(N) .< 0.01)
        yn = y + e    # Measurement signal with noise

        d = iddata(yn, Δt)
        model = arma(d, na, nc, initial_order = 20)

        @test numvec(model)[1] ≈ numvec(G)[1] atol = 0.5
        @test denvec(model)[1] ≈ denvec(G)[1] atol = 0.5
        @test freqresptest(G, model) < 0.3

        uh = ControlSystemIdentification.estimate_residuals(model, yn)
        @show mean(abs2, uh - u) / mean(abs2, u)
        @test mean(abs2, uh - u) / mean(abs2, u) < 0.01

    end

    @testset "arma ssa" begin
        @info "Testing arma ssa"

        T = 1000
        G = tf(1, [1, 2 * 0.1 * 1, 1])
        G = c2d(G, 1)
        u = randn(T)
        y = lsim(G, u, 1:T)[1][:]
        d = iddata(y)
        model = ControlSystemIdentification.arma_ssa(d, 2, 2, L = 200)

        @test numvec(model)[1] ≈ numvec(G)[1] atol = 0.7
        @test denvec(model)[1] ≈ denvec(G)[1] atol = 0.2

        @test freqresptest(G, model) < 2
    end




    # end
    @testset "frd" begin
        Random.seed!(1)
        ##
        T           = 100000
        h           = 1
        t           = range(0, step = h, length = T)
        sim(sys, u) = lsim(sys, u, t)[1][:]
        σy          = 0.5
        sys         = tf(1, [1, 2 * 0.1, 0.1])
        ωn          = sqrt(0.3)
        sysn        = tf(σy * ωn, [1, 2 * 0.1 * ωn, ωn^2])

        u  = randn(T)
        y  = sim(sys, u)
        yn = y + sim(sysn, randn(size(u)))
        d  = iddata(y, u, h)
        dn = iddata(yn, u, 1)

        # using BenchmarkTools
        # @btime begin
        # Random.seed!(0)
        k = coherence(d)
        @test all(k.r .> 0.99)
        k = coherence(dn)
        @test all(k.r[1:10] .> 0.9)
        @test k.r[end] .> 0.8
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

    end


    @testset "plots and difficult" begin

        Random.seed!(1)
        ##
        T = 200
        nx = 2
        nu = 1
        ny = 1
        x0 = randn(nx)
        σy = 0.5
        sim(sys, u) = lsim(sys, u', 1:T)[1]'
        sys = tf(1, [1, 2 * 0.1, 0.1])
        sysn = tf(σy, [1, 2 * 0.1, 0.3])

        u  = randn(nu, T)
        un = u + 0.1randn(size(u))
        y  = sim(sys, u)
        yn = y + sim(sysn, σy * randn(size(u)))
        dd = iddata(yn, un, 1)

        uv  = randn(nu, T)
        yv  = sim(sys, uv)
        ynv = yv + sim(sysn, σy * randn(size(uv)))
        dv  = iddata(yv, uv, 1)
        dnv = iddata(ynv, uv, 1)
        ##

        res = [
            pem(dnv, nx = nx, iterations = 1000, difficult = true, focus = :prediction)
            for nx in [1, 3, 4]
        ]

        ω = exp10.(range(-2, stop = log10(pi), length = 150))
        fig = plot(layout = 4, size = (1000, 600))
        for i in eachindex(res)
            (sysh, x0h, opt) = res[i]
            ControlSystemIdentification.simplot!(
                sysh,
                dnv,
                x0h;
                subplot = 1,
                ploty = i == 1,
            )
            ControlSystemIdentification.predplot!(
                sysh,
                dnv,
                x0h;
                subplot = 2,
                ploty = i == 1,
            )
        end
        bodeplot!(
            ss.(getindex.(res, 1)),
            ω,
            plotphase = false,
            subplot = 3,
            title = "Process",
            linewidth = 2 * [4 3 2 1],
        )
        bodeplot!(
            ControlSystems.innovation_form.(getindex.(res, 1)),
            ω,
            plotphase = false,
            subplot = 4,
            linewidth = 2 * [4 3 2 1],
        )
        bodeplot!(
            sys,
            ω,
            plotphase = false,
            subplot = 3,
            lab = "True",
            linecolor = :blue,
            l = :dash,
            legend = :bottomleft,
            title = "System model",
        )
        bodeplot!(
            ControlSystems.innovation_form(ss(sys), syse = ss(sysn), R2 = σy^2 * I),
            ω,
            plotphase = false,
            subplot = 4,
            lab = "True",
            linecolor = :blue,
            l = :dash,
            ylims = (0.1, 100),
            legend = :bottomleft,
            title = "Noise model",
        )
        display(fig)

    end


    @testset "impulseest" begin
        T = 200
        h = 0.1
        t = h:h:T
        sim(sys, u) = lsim(sys, u, t)[1][:]
        sys = c2d(tf(1, [1, 2 * 0.1, 0.1]), h)

        u = randn(length(t))
        y = sim(sys, u) + 0.1randn(length(t))

        d = iddata(y, u, h)
        impulseestplot(d, Int(50 / h), λ = 0)
        impulseplot!(sys, 50, l = (:dash, :blue))
    end

    @testset "Spectrogram" begin
        @info "Testing Spectrogram"

        T = 1000
        s = sin.((1:T) .* 2pi / 10)


        S1 = spectrogram(s, window = hanning)

        estimator = model_spectrum(ar, 1, 2)
        S2 = spectrogram(s, estimator, window = hanning)
        @test maximum(
            findmax(S1.power, dims = 1)[2][:] - findmax(S2.power, dims = 1)[2][:],
        ) <= CartesianIndex(1, 0)
        @test minimum(
            findmax(S1.power, dims = 1)[2][:] - findmax(S2.power, dims = 1)[2][:],
        ) >= CartesianIndex(-1, 0)

        estimator = model_spectrum(arma, 1, 2, 1)
        S2 = spectrogram(s, estimator, window = hanning)
        @test maximum(
            findmax(S1.power, dims = 1)[2][:] - findmax(S2.power, dims = 1)[2][:],
        ) <= CartesianIndex(1, 0)
        @test minimum(
            findmax(S1.power, dims = 1)[2][:] - findmax(S2.power, dims = 1)[2][:],
        ) >= CartesianIndex(-1, 0)
    end
end





##
# using TotalLeastSquares
# using LinearAlgebra: QRIteration
# using Random
# plotly(size=(600,400))
# N = 500
# ivec = 1:5:50
# res = tmap(ivec) do i
#     Random.seed!(i)
#     r = 10
#     A = Matrix{Float64}(I(r))
#     A[1, 1] = 1.01
#     G = ss(A, randn(r, m), randn(l, r), 0 * randn(l, m), 1)
#     r = G.nx
#     res = map(1:20) do _
#         u = randn(N, G.nu)
#         x0 = randn(r)
#         y, t, x = lsim(G, u, 1:N, x0 = x0)
#         @test sum(!isfinite, y) == 0
#         yn = y + 0.1randn(size(y))
#         d = iddata(yn, u, 1)
#
#         # res = n4sid(d, r, i = r+i, svd = x->svd(x, alg = QRIteration()))
#         res = n4sid2(d, r, i = r+i)
#         # freqresptest(G, res.sys) < 0.2 * m * l
#         w = exp10.(LinRange(-2, log10(pi), 600))
#         b1,_ = bode(G, w)
#         b2,_ = bode(res.sys, w)
#         mean(abs2, log.(b1)-log.(b2))
#     end
#     mean(res)
# end
#
# plot(ivec, res)
