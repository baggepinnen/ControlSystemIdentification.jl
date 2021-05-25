wtest = exp10.(LinRange(-3, log10(pi), 30))
freqresptest(G, model) =
    maximum(abs, log10.(abs2.(freqresp(model, wtest))) - log10.(abs2.(freqresp(G, wtest))))

freqresptest(G, model, tol) = freqresptest(G, model) < tol

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
        u = randn(m, N)
        x0 = randn(r)
        y, t, x = lsim(G, u, 1:N, x0 = x0)
        @test sum(!isfinite, y) == 0
        yn = y + 0.1randn(size(y))
        d = iddata(yn, u, 1)
        res = n4sid(d, r, γ = 0.99)
        @test maximum(abs, pole(res.sys)) <= 1.00001 * 0.99

        ys = simulate(res, d)
        # @show mean(abs2,y-ys') / mean(abs2,y)
        ys = simulate(res, d, stochastic = true)
        # @show mean(abs2,y-ys') / mean(abs2,y)

        yp = predict(res, d)
        # @show mean(abs2,y-yp') / mean(abs2,y)
        @test mean(abs2, y - yp) / mean(abs2, y) < 0.01


        res = n4sid(d, r, i = r + 1)
        freqresptest(G, res.sys) < 0.2 * m * l
        w = exp10.(LinRange(-5, log10(pi), 600))
        bodeplot(G, w)
        bodeplot!(res.sys, w)


        G = ss(0.2randn(r, r), randn(r, m), randn(l, r), 0 * randn(l, m), 1)
        u = randn(m, N)

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
            @test norm(iy - permutedims(H, (1, 3, 2))) < 0.05
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
        u = randn(m, N)
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

@testset "subspaceid" begin
    @info "Testing subspaceid"
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
        u = randn(m, N)
        x0 = randn(r)
        y, t, x = lsim(G, u, 1:N, x0 = x0)
        @test sum(!isfinite, y) == 0
        yn = y + 0.1randn(size(y))
        d = iddata(yn, u, 1)
        res = subspaceid(d, r)

        ys = simulate(res, d)
        # @show mean(abs2,y-ys) / mean(abs2,y)
        ys = simulate(res, d, stochastic = true)
        # @show mean(abs2,y-ys) / mean(abs2,y)

        yp = predict(res, d)
        # @show mean(abs2,y-yp') / mean(abs2,y)
        @test mean(abs2, y - yp) / mean(abs2, y) < 0.1 # no enforcement of stability in this method


        res = subspaceid(d, r)
        freqresptest(G, res.sys) < 0.2 * m * l
        w = exp10.(LinRange(-5, log10(pi), 600))
        bodeplot(G, w)
        bodeplot!(res.sys, w)


        G = ss(0.2randn(r, r), randn(r, m), randn(l, r), 0 * randn(l, m), 1)
        u = randn(m, N)

        y, t, x = lsim(G, u, 1:N, x0 = randn(r))
        @assert sum(!isfinite, y) == 0
        ϵ = 0.01
        yn = y + ϵ * randn(size(y))
        d = iddata(yn, u, 1)

        res = subspaceid(d, r)
        @test res.sys.nx == r
        @test norm(res.S) < 2ϵ
        @test freqresptest(G, res.sys) < 0.2 * m * l


        res = ControlSystemIdentification.subspaceid(d)
        @test res.sys.nx <= r # test that auto rank selection don't choose too high rank when noise is low
        kf = KalmanFilter(res)
        @test kf isa KalmanFilter
        @test kf.A == res.A
        @test kf.B == res.B
        @test kf.C == res.C

    end

    m = 1; l = 1; r = 1
    a = -0.9; b = 1
    for a in LinRange(-0.99, 0.99, 10), b in LinRange(-5, 5, 10)
        G = tf(b, [1, -a], 1)
        u = randn(m, N)
        y, t, x = lsim(G, u, 1:N, x0 = randn(r))
        yn = y + 0.01randn(size(y))
        d = iddata(yn, u, 1)
        res = subspaceid(d, r, r=20, W=:MOESP, zeroD=true)
        @test res.sys.A[1] ≈ a atol = 0.01
        @test numvec(tf(res.sys))[1][end] ≈ b atol = 0.01 # might be bias due to no initial state when estimating B/D
        # @test abs(numvec(tf(res.sys))[1][1]) < 1e-2
        @test freqresptest(G, res.sys) < 0.01
    end
end
