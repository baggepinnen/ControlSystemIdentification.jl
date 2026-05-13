using ControlSystemIdentification, ControlSystemsBase
using Test, Random, LinearAlgebra, Statistics
using ComponentArrays

# Simulate from a known affine-in-λ LPV system. Returns (d, λ, sys_true_factory).
function simulate_affine_lpv(; T = 2000, Ts = 0.05, σy = 0.01, seed = 1)
    Random.seed!(seed)
    A0 = [0.85 0.10; -0.05 0.80]
    A1 = [0.05 0.00;  0.10 -0.05]
    B0 = reshape([0.10, 0.20], 2, 1)
    B1 = reshape([0.00, 0.05], 2, 1)
    C  = [1.0 0.0]
    A(λ) = A0 .+ λ .* A1
    B(λ) = B0 .+ λ .* B1
    λ_traj = 0.5 .* sin.(0.01 .* (1:T))
    u = randn(1, T)
    x = zeros(2)
    y = zeros(1, T)
    for t in 1:T
        y[:, t] = C * x
        x = A(λ_traj[t]) * x + B(λ_traj[t]) * u[:, t]
    end
    y_meas = y .+ σy .* randn(size(y))
    d = iddata(y_meas, u, Ts)
    return d, λ_traj, (; A0, A1, B0, B1, C, Ts)
end

@testset "LPVStateSpace basic" begin
    # Construct manually and check frozen evaluation
    nx, nu, ny, nb = 2, 1, 1, 2
    A_arr = randn(nx, nx, nb)
    B_arr = randn(nx, nu, nb)
    C_arr = randn(ny, nx, nb)
    D_arr = zeros(ny, nu, nb)
    θ = ComponentArray(A = A_arr, B = B_arr, C = C_arr, D = D_arr)
    K = zeros(nx, ny)
    basis = λ -> [1.0, λ]
    sys = LPVStateSpace(basis, nb, θ, K, 0.1, nx, nu, ny)
    s0 = sys(0.0)
    @test s0.A ≈ A_arr[:, :, 1]
    @test s0.B ≈ B_arr[:, :, 1]
    @test s0.C ≈ C_arr[:, :, 1]
    s1 = sys(1.0)
    @test s1.A ≈ A_arr[:, :, 1] .+ A_arr[:, :, 2]
end

@testset "LPV simulate/predict shapes" begin
    nx, nu, ny, nb = 2, 1, 1, 2
    θ = ComponentArray(
        A = cat([0.9 0.1; 0.0 0.8], zeros(2, 2); dims = 3),
        B = cat(reshape([0.1, 0.2], 2, 1), zeros(2, 1); dims = 3),
        C = cat([1.0 0.0], zeros(1, 2); dims = 3),
        D = zeros(1, 1, 2),
    )
    sys = LPVStateSpace(λ -> [1.0, λ], nb, θ, zeros(nx, ny), 0.05, nx, nu, ny)
    T = 100
    u = randn(1, T)
    λ = 0.1 .* sin.(1:T)
    y = ControlSystemIdentification.simulate(sys, u, λ)
    @test size(y) == (ny, T)
    d = iddata(y, u, sys.Ts)
    yh = ControlSystemIdentification.predict(sys, d, λ)
    @test size(yh) == (ny, T)
end

@testset "LPV warmstart returns the right shapes" begin
    d, λ, _ = simulate_affine_lpv(T = 1500)
    θ0 = lpv_warmstart(d, λ, 2; basis = [_ -> 1.0, λ -> λ])
    @test size(θ0.A) == (2, 2, 2)
    @test size(θ0.B) == (2, 1, 2)
    @test size(θ0.C) == (1, 2, 2)
    @test size(θ0.D) == (1, 1, 2)
    @test all(isfinite, θ0.A)
    @test all(isfinite, θ0.B)
    @test all(isfinite, θ0.C)
end

@testset "LPV PEM recovery beats LTI on varying-λ data" begin
    d, λ, _ = simulate_affine_lpv(T = 2000, σy = 0.005)
    basis = [_ -> 1.0, λ -> λ]

    # Single-LTI baseline
    sys_lti = subspaceid(d, 2)
    yh_lti = predict(sys_lti, d)
    e_lti = mean(abs2, d.y .- yh_lti)

    # LPV fit. K0 is pinned for determinism across Julia/Random versions.
    res = lpv_pem(d, λ, 2; basis,
                  K0 = 1e-6 .* ones(2, 1),
                  show_trace = false, store_trace = false,
                  iterations = 200, time_limit = 60)
    sys_lpv, x0h, _ = res

    yh_lpv = ControlSystemIdentification.predict(sys_lpv, d, λ; x0 = x0h)
    e_lpv = mean(abs2, d.y .- yh_lpv)

    @test e_lpv < e_lti  # LPV should be strictly better on truly varying-λ data
    @test e_lpv < 0.05   # Coarse absolute bound — adjust if seed/tolerances change
end

@testset "LPV PEM with zeroD" begin
    d, λ, _ = simulate_affine_lpv(T = 1500, σy = 0.005)
    basis = [_ -> 1.0, λ -> λ]
    res = lpv_pem(d, λ, 2; basis, zeroD = true,
                  K0 = 1e-6 .* ones(2, 1),
                  show_trace = false, store_trace = false,
                  iterations = 100, time_limit = 60)
    sys_lpv, _, _ = res
    @test all(iszero, sys_lpv.θ.D)
    # Predict should still work
    yh = ControlSystemIdentification.predict(sys_lpv, d, λ)
    @test size(yh) == (1, length(d))
end

@testset "LPV PEM basis-of-length-1 ≈ LTI" begin
    # When the basis has a single constant function, the LPV model is just LTI;
    # the result should match a plain LTI fit to within a moderate tolerance.
    Random.seed!(2)
    Ts = 0.05
    A = [0.85 0.10; -0.05 0.80]
    B = reshape([0.10, 0.20], 2, 1)
    C = [1.0 0.0]
    sys_true = ss(A, B, C, 0, Ts)
    T = 1500
    u = randn(1, T)
    y, _, _ = lsim(sys_true, u)
    y .+= 0.005 .* randn(size(y))
    d = iddata(y, u, Ts)
    λ = zeros(T)

    res = lpv_pem(d, λ, 2; basis = [_ -> 1.0],
                  K0 = 1e-6 .* ones(2, 1),
                  show_trace = false, store_trace = false,
                  iterations = 200, time_limit = 60)
    sys_lpv, x0h, _ = res
    yh = ControlSystemIdentification.predict(sys_lpv, d, λ; x0 = x0h)
    e = mean(abs2, d.y .- yh)
    @test e < 0.01
end
