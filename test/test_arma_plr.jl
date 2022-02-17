@testset "plr" begin

    N = 2000
    t = 1:N
    u = randn(1,N)
    G = tf(0.8, [1, -0.9], 1)
    y = lsim(G, u, t)[1]
    e = randn(1,N)
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
    @test freqresptest(G, Gplr) < 0.11

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
    u  = randn(1,N) # A random control input
    a  = 0.9

    G = tf([1, 0.1], [1, -a, 0], 1)
    y, t, x = lsim(G, u, 1:N) 

    na, nc = 2, 2   # Number of polynomial coefficients
    e = 0.0001randn(1,N) #+ 20randn(1,N) .* (rand(N) .< 0.01)
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
    u = randn(1,T)
    y = lsim(G, u, 1:T)[1]
    d = iddata(y)
    model = ControlSystemIdentification.arma_ssa(d, 2, 2, L = 200)

    @test numvec(model)[1] ≈ numvec(G)[1] atol = 0.7
    @test denvec(model)[1] ≈ denvec(G)[1] atol = 0.2

    @test freqresptest(G, model) < 2
end