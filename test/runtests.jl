if haskey(ENV, "CI")
    ENV["PLOTS_TEST"] = "true"
    ENV["GKSwstype"] = "100" # gr segfault workaround
end

using ControlSystemIdentification, ControlSystemsBase, Optim, Plots, TotalLeastSquares
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

    @testset "prediction" begin
        @info "Testing prediction"
        include("test_prediction.jl")
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
        @info "Testing pem"
        include("test_pem.jl")
    end

    @testset "nonlinear_pem" begin
        @info "Testing nonlinear_pem"
        include("test_nonlinear_pem.jl")
    end

    @testset "arx" begin
        @info "Testing arx"
        include("test_arx.jl")
    end

    @testset "ar" begin
        @info "Testing ar"
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
        Random.seed!(1)
        N = 200
        r = 2
        m = 2
        l = 2
        A = Matrix{Float64}(I(r))
        A[1, 1] = 1.01
        Ts = 0.5
        G = ss(A, randn(r, m), randn(l, r), 0 * randn(l, m), Ts)
        u = randn(m, N)
        x0 = randn(r)
        y, t, x = lsim(G, u, x0 = x0)
        @test sum(!isfinite, y) == 0
        yn = y + 0.1randn(size(y))
        d = iddata(yn, u, Ts)
        sysd = n4sid(d, r, γ = 0.99)
        Qc = d2c(sysd, sysd.Q, opt=:o)
        Qd = c2d(sysd, Qc, opt=:o)
        @test Qd ≈ sysd.Q

        sysc = d2c(sysd)
        @test sysc.Q ≈ Qc
        Qd2 = c2d(sysc, Qc, sysd.Ts, opt=:o)
        @test Qd2 ≈ Qd rtol=1e-5

        sysd2 = c2d(sysc, sysd.Ts)
        @test sysd2.Q ≈ sysd.Q
        @test sysd2.K ≈ sysd.K rtol=1e-5

        # Test sampling of cost matrix
        sys = DemoSystems.resonant()
        x0 = ones(sys.nx)

        Ts = 0.01 # cost approximation becomes more crude as Ts increases, expected?
        Qc = [1 0.01; 0.01 2]
        Rc = I(1)
        sysd = c2d(sys, Ts)
        Qd, Rd = c2d(sys, Qc, Rc, Ts, opt=:c)
        Qc2 = d2c(sysd, Qd; opt=:c)
        @test Qc2 ≈ Qc

        # test case from paper
        A = [
            2 -8 -6
            10 -19 -12
            -10 15 8
        ]
        B = [
            5 1
            1 4
            3 2
        ]
        Qc = [
            4 1 2
            1 3 1
            2 1 5
        ]

        Qd = c2d(ss(A,B,I,0), Qc, 1, opt=:c)
        Qd_van_load = [
            9.934877720 -11.08568953 -9.123023900
            -11.08568953 13.66870748 11.50451512
            -9.123023900 11.50451512 10.29179555 
        ]

        @test norm(Qd - Qd_van_load) < 1e-6


        # NOTE: these tests are not run due to OrdinaryDiffEq latency, they should pass
        # using OrdinaryDiffEq
        # L = lqr(sys, Qc, Rc)
        # dynamics = function (xc, p, t)
        #     x = xc[1:sys.nx]
        #     u = -L*x
        #     dx = sys.A*x + sys.B*u
        #     dc = dot(x, Qc, x) + dot(u, Rc, u)
        #     return [dx; dc]
        # end
        # prob = ODEProblem(dynamics, [x0; 0], (0.0, 10.0))
        # sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
        # cc = sol.u[end][end]
        # Ld = lqr(sysd, Qd, Rd)
        # sold = lsim(sysd, (x, t) -> -Ld*x, 0:Ts:10, x0 = x0)
        # function cost(x, u, Q, R)
        #     dot(x, Q, x) + dot(u, R, u)
        # end
        # cd = cost(sold.x, sold.u, Qd, Rd)
        # @test cc ≈ cd rtol=0.01
        # @test abs(cc-cd) < 1.0001*0.005531389319983315

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





