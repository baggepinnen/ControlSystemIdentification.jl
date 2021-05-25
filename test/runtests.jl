using ControlSystemIdentification, ControlSystems, Optim, Plots, DSP, TotalLeastSquares
using Test, Random, LinearAlgebra, Statistics
import ControlSystemIdentification as CSI

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

##

@testset "ControlSystemIdentification.jl" begin

    @testset "estimate x0" begin
        @info "Testing estimate x0"
        include("test_estimatex0.jl")
    end

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
        include("test_iddata.jl")
    end

    @testset "Subspace" begin
        @info "Testing Subspace"
        include("test_subspace.jl")
    end

    @testset "pem" begin
        include("test_pem.jl")
    end

    @testset "arx" begin
        include("test_arx.jl")
    end

    @testset "ar" begin
        include("test_ar.jl")
    end

    @testset "arxar" begin
        @info "Testing arxar"
        include("test_arxar.jl")
    end
    
    @testset "arma_plr" begin
        @info "Testing arma_plr"
        include("test_arma_plr.jl")
    end

    @testset "frd" begin
        @info "Testing frd"
        include("test_frd.jl")
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
            noise_model.(getindex.(res, 1)),
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
        sim(sys, u) = lsim(sys, u, t)[1]
        sys = c2d(tf(1, [1, 2 * 0.1, 0.1]), h)

        u = randn(1,length(t))
        y = sim(sys, u) + 0.1randn(1,length(t))

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

    @testset "sampling of covariance matrices" begin
        @info "Testing sampling of covariance matrices"
        N = 200
        r = 2
        m = 2
        l = 2
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
        Qc = d2c(res.sys, res.Q)
        Qd = c2d(res.sys, Qc)
        @test Qd ≈ res.Q
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
