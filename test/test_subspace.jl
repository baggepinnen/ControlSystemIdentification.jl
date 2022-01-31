using ControlSystemIdentification, ControlSystems
using ControlSystems: isdiscrete
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
        G = ss(A, ones(r, m), ones(l, r), 0 * ones(l, m), 1)
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
            @test norm(iy - permutedims(H, (1, 3, 2))) < 0.06
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
    nx = 2
    m = 2
    l = 2
    for nx = 1:5, m = 1:2, l = 1:2
        Random.seed!(0)
        @show nx, m, l
        A = Matrix{Float64}(I(nx))
        A[1, 1] = 1.01
        G = ss(A, ones(nx, m), ones(l, nx), 0 * ones(l, m), 1)
        u = randn(m, N)
        x0 = randn(nx)
        y, t, x = lsim(G, u, 1:N, x0 = x0)
        @test sum(!isfinite, y) == 0
        yn = y + 0.1randn(size(y))
        d = iddata(yn, u, 1)
        res = subspaceid(d, nx, focus=:simulation)

        ys = simulate(res, d)
        # @show mean(abs2,y-ys) / mean(abs2,y)
        ys = simulate(res, d, stochastic = true)
        # @show mean(abs2,y-ys) / mean(abs2,y)

        yp = predict(res, d)
        # @show mean(abs2,y-yp') / mean(abs2,y)
        @test mean(abs2, y - yp) / mean(abs2, y) < 0.1 # no enforcement of stability in this method

        freqresptest(G, res.sys) < 0.2 * m * l
        w = exp10.(LinRange(-5, log10(pi), 600))
        bodeplot(G, w)
        bodeplot!(res.sys, w)


        G = ss(0.2randn(nx, nx), randn(nx, m), randn(l, nx), 0 * randn(l, m), 1)
        u = randn(m, N)

        y, t, x = lsim(G, u, 1:N, x0 = randn(nx))
        @assert sum(!isfinite, y) == 0
        ϵ = 0.01
        yn = y + ϵ * randn(size(y))
        d = iddata(yn, u, 1)

        res = subspaceid(d, nx)
        @test res.sys.nx == nx
        @test norm(res.S) < 2ϵ
        @test freqresptest(G, res.sys) < 0.2 * m * l


        res = ControlSystemIdentification.subspaceid(d)
        @test res.sys.nx <= nx # test that auto rank selection don't choose too high rank when noise is low
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



@testset "Balreal" begin
    @info "Testing Balreal"
    m = 1
    k = 100
    c = 1
    G = ss(tf(k/m, [1, c/m, k/m]))
    Q,R = [1 0.1; 0.1 2], I(1)
    K = kalman(G, Q, R) |> real
    sys = ControlSystemIdentification.PredictionStateSpace(G, K, Q, R)

    function cl_eig(G, K)
        A,B,C,D = ssdata(G)
        eigvals(A - K*C)
    end
    e0 = cl_eig(G, K)

    # test that the transformation of K leaves the closed-loop poles intact
    Gb, gra, T = balreal(sys)
    @test cl_eig(Gb, Gb.K) ≈ e0 rtol=1e-6
    @test iscontinuous(Gb)

    # test that the transformation of Q is correct
    Kb = kalman(Gb.sys, Gb.Q, Gb.R)
    @test cl_eig(Gb, Kb) ≈ e0 rtol=1e-6

    Gr = baltrunc(sys, n=1)[1]
    @test Gr.nx == 1
    @test size(Gr.K) == (1,1)


    T = randn(2,2)
    syst = similarity_transform(sys, T)
    @test iscontinuous(syst)
    @test iscontinuous(syst.sys)
    @test tf(noise_model(sys)) ≈ tf(noise_model(syst))

    Kt = kalman(syst.sys, syst.Q, syst.R)
    @test Kt ≈ syst.K
    @test cl_eig(syst, Kt) ≈ e0 rtol=1e-6


    # Discrete
    m = 1
    G = c2d(ss(tf(k/m, [1, c/m, k/m])), 0.1)
    Q,R = [1 0.1; 0.1 2], I(1)
    K = kalman(G, Q, R) |> real
    e0 = cl_eig(G, K)
    sys = ControlSystemIdentification.PredictionStateSpace(G, K, Q, R)
    @test isdiscrete(sys)
    Gb, gra, T = balreal(sys)
    @test isdiscrete(Gb)
    @test Gb.Ts == 0.1

    # Test that the kalman gain is correctly transformed
    @test tf(noise_model(sys)) ≈ tf(noise_model(Gb))

        # test that the transformation of Q is correct
    Kb = kalman(Gb.sys, Gb.Q, Gb.R)
    @test cl_eig(Gb, Kb) ≈ e0 rtol=1e-6

    T = [1 0.1; 0.1 1]#randn(2,2)
    syst = similarity_transform(sys, T)
    @test isdiscrete(syst)
    @test isdiscrete(syst.sys)
    @test tf(noise_model(sys)) ≈ tf(noise_model(syst))

    Kt = kalman(syst.sys, syst.Q, syst.R)
    @test Kt ≈ syst.K
    @test cl_eig(syst, Kt) ≈ e0 rtol=1e-6

end

@testset "similarity transform" begin
    @info "Testing similarity transform"
    T = randn(3,3)
    sys1 = ssrand(1,1,3)
    sys2 = ControlSystems.similarity_transform(sys1, T)
    T2 = ControlSystemIdentification.find_similarity_transform(sys1, sys2)
    @test T2 ≈ T atol=1e-8

    T3 = ControlSystemIdentification.find_similarity_transform(sys1, sys2, :ctrb)
    @test T3 ≈ T atol=1e-8

end

@testset "schur_stab" begin
    a = diagm([1.1, 0.999, 0.5, 0])
    Q = randn(4,4)
    A = Q\a*Q
    As = ControlSystemIdentification.schur_stab(A)
    @test all(abs.(eigvals(As)) .< 1)
    @test eigvals(As) ≈ reverse([0.99, 0.9, 0.5, 0]) atol=1e-8
end


## Freq domain
@testset "freq domain" begin
    @info "Testing subspaceid freq domain"

    N = 200
    ny,nu,nx = 2,3,4
    u = randn(nu, N)

    # BD/CD
    zeroD = false
    for zeroD = (true, false)
        @show zeroD
        G = ssrand(ny,nu,nx, Ts=1, proper=zeroD)
        y, t, x = ControlSystems.lsim(G, u)

        w = exp10.(LinRange(-4, log10(2pi*0.5), 200))
        F = freqresp(G, w).parent
        Y, U, Ω = ControlSystemIdentification.ifreqresp(F, w)

        for j = 1:N
            for m = 0:nu-1
                @test Y[:,j+m*N] ≈ F[:,:,j]*U[:,j+m*N]
            end
        end

        λ = cis.(Ω)

        C,D,e = ControlSystemIdentification.find_CDf(G.A, G.B, U, Y, λ, zeros(G.nx), zeroD, \, false)

        @test sum(abs2, e) < eps()
        @test C ≈ G.C
        @test D ≈ G.D

        B,D,x0,e = ControlSystemIdentification.find_BDf(G.A, G.C, U, Y, λ, zeroD, \, false)
        
        @test sum(abs2, e) < eps()
        @test iszero(x0)
        @test B ≈ G.B
        @test D ≈ G.D

        Y = ControlSystemIdentification.fft(y, 2)
        U = ControlSystemIdentification.fft(u, 2)

        frd = FRD(w, F)
        df = ControlSystemIdentification.ifreqresp(frd)
        
        Gh, x0 = subspaceid(frd, G.Ts, nx; r=2nx, zeroD, verbose=true)
        @test hinfnorm(G-Gh)[1] < 1e-12

        d = iddata(y, u, G.Ts)
        df = ControlSystemIdentification.fft(d)
        @test df.Ts == 1

        Gh, x0 = subspaceid(df, df.Ts, nx; r=2nx, estimate_x0=true, zeroD)
        @test hinfnorm(G-Gh)[1] < 1e-12


    end

    # Weights
    N = 200
    ny,nu,nx = 3,2,3
    u = randn(nu, N)
    G = ssrand(ny,nu,nx, Ts=1, proper=zeroD)
    y, t, x = ControlSystems.lsim(G, u)
    
    w = exp10.(LinRange(-4, log10(2pi*0.5), 200))
    F = freqresp(G, w).parent
    F = F .+ 1 .* randn.(ComplexF64)
    
    frd = FRD(w, F)
    Gh,_ = ControlSystemIdentification.subspaceid(frd, G.Ts, nx; r=5nx, zeroD, verbose=true)
    
    weights = [ones(100); 10ones(100)]
    Ghw,_ = ControlSystemIdentification.subspaceid(frd, G.Ts, nx; r=5nx, zeroD, verbose=true, weights)
    
    isinteractive() && bodeplot([G, Gh, Ghw], w, plotphase=false)

end

