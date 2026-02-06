using ControlSystemIdentification, ControlSystemsBase
using Test, Random, LinearAlgebra, Statistics
using ControlSystemIdentification: pbsid

@testset "pbsid" begin
    @info "Testing pbsid"

    @testset "SISO identification" begin
        Random.seed!(42)
        a = 0.9
        b = 1.0
        G = tf(b, [1, -a], 1)
        nx = 1

        N = 500
        u = randn(1, N)
        y, _, _ = lsim(G, u, 1:N)
        yn = y + 0.01 * randn(size(y))

        d = iddata(yn, u, 1)
        res = pbsid(d, nx; f=10, p=10, verbose=false)

        @test res.sys.nx == nx

        # Check poles match
        true_pole = a
        est_pole = res.sys.A[1]
        @test est_pole ≈ true_pole atol=0.05

        # Check prediction performance
        yp = predict(res, d)
        vaf = 1 - sum(abs2, yn - yp) / sum(abs2, yn .- mean(yn))
        @test vaf > 0.99
    end

    @testset "MIMO identification" begin
        Random.seed!(123)

        nx = 2
        ny = 2
        nu = 2
        G = ssrand(ny, nu, nx, Ts=1, proper=true)

        # Generate data with white noise input
        N = 1000
        u = randn(nu, N)
        y, _, _ = lsim(G, u, 1:N)
        yn = y + 0.01 * randn(size(y))

        d = iddata(yn, u, 1)

        # Identify with pbsid
        res = pbsid(d, nx; f=10, p=10, verbose=false)

        @test res.sys.nx == nx

        # Check that the identified system is stable
        @test all(abs.(eigvals(res.sys.A)) .<= 1.0 + 1e-6)

        # Check prediction performance
        yp = predict(res, d)
        total_vaf = 1 - sum(abs2, yn - yp) / sum(abs2, yn .- mean(yn, dims=2))
        @test total_vaf > 0.99
    end

    @testset "weight=0 vs weight=1" begin
        Random.seed!(456)

        nx = 2
        G = ssrand(1, 1, nx, Ts=1, proper=true)
        G = G / norm(G, Inf)  # Normalize

        N = 500
        u = randn(1, N)
        y, _, _ = lsim(G, u, 1:N)
        yn = y + 0.01 * randn(size(y))

        d = iddata(yn, u, 1)

        res0 = pbsid(d, nx; f=10, p=10, weight=0, verbose=false)
        res1 = pbsid(d, nx; f=10, p=10, weight=1, verbose=false)

        # Both should give good prediction
        yp0 = predict(res0, d)
        yp1 = predict(res1, d)

        vaf0 = 1 - sum(abs2, yn - yp0) / sum(abs2, yn .- mean(yn))
        vaf1 = 1 - sum(abs2, yn - yp1) / sum(abs2, yn .- mean(yn))

        @test vaf0 > 0.99
        @test vaf1 > 0.99
    end

    @testset "automatic order selection" begin
        Random.seed!(789)

        nx = 3
        G = ssrand(1, 1, nx, Ts=1, proper=true)
        G = G / norm(G, Inf)

        N = 500
        u = randn(1, N)
        y, _, _ = lsim(G, u, 1:N)
        yn = y + 0.001 * randn(size(y))  # Low noise

        d = iddata(yn, u, 1)

        res = pbsid(d, :auto; f=15, p=15, verbose=false)

        # Should select a positive order
        @test res.sys.nx >= 1
        # Check that the model is stable
        @test all(abs.(eigvals(res.sys.A)) .<= 1.0 + 1e-6)
    end

end
