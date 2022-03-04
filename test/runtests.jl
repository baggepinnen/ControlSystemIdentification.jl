if haskey(ENV, "CI")
    ENV["PLOTS_TEST"] = "true"
    ENV["GKSwstype"] = "100" # gr segfault workaround
end

using ControlSystemIdentification, ControlSystems, Optim, Plots, TotalLeastSquares
import DSP
using DSP: spectrogram, hanning
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
    maximum(abs, log10.(abs2.(freqresp(model, wtest))) .- log10.(abs2.(freqresp(G, wtest))))

freqresptest(G, model, tol) = freqresptest(G, model) < tol

"""
Compare a tf to a tf based on particles .. maybe replaye by correct dispatched isapprox
"""
function compareTFs(tf, tfStochastic, atol = 10e-5)
    aok = bok = false
    try
        bok = isapprox(numvec(tf), pmean.(numvec(tfStochastic)), atol = atol)
        aok = isapprox(denvec(tf), pmean.(denvec(tfStochastic)), atol = atol)
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
        include("test_utils.jl")
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

    @testset "plots" begin
        @info "Testing plots"
        include("test_plots.jl")
    end

    @testset "simplots and difficult" begin
        @info "Testing simplots and difficult"
        include("test_simplots_difficult.jl")
    end

    @testset "impulseest" begin
        @info "Testing impulseest"
        include("test_impulseest.jl")
    end

    @testset "Spectrogram" begin
        @info "Testing Spectrogram"
        T = 1000
        s = sin.((1:T) .* 2pi / 10)
        S1 = spectrogram(s, window = DSP.hanning)

        estimator = model_spectrum(ar, 1, 2)
        S2 = spectrogram(s, estimator, window = DSP.hanning)
        @test maximum(
            findmax(S1.power, dims = 1)[2][:] - findmax(S2.power, dims = 1)[2][:],
        ) <= CartesianIndex(1, 0)
        @test minimum(
            findmax(S1.power, dims = 1)[2][:] - findmax(S2.power, dims = 1)[2][:],
        ) >= CartesianIndex(-1, 0)

        estimator = model_spectrum(arma, 1, 2, 1)
        S2 = spectrogram(s, estimator, window = DSP.hanning)
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
        sys = n4sid(d, r, γ = 0.99)
        Qc = d2c(sys, sys.Q)
        Qd = c2d(sys, Qc)
        @test Qd ≈ sys.Q

        sysc = d2c(sys)
        sysd = c2d(sysc, sys.Ts)
        @test sysd.Q ≈ sys.Q
        @test sysd.K ≈ sys.K rtol=1e-2
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
