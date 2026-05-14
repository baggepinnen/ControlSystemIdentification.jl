using ControlSystemIdentification, SeeToDee, LowLevelParticleFilters
using LeastSquaresOptim
using StaticArrays
using LinearAlgebra
using Random, Statistics
using Test

function quadtank(h, u, p, t)
    k1, k2, g = p[1], p[2], 9.81
    A1 = A3 = A2 = A4 = p[3]
    a1 = a3 = a2 = a4 = 0.03
    γ1 = γ2 = p[4]

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0

    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end
measurement(x,u,p,t) = SA[x[1], x[2]]

Ts = 1.0
discrete_dynamics = SeeToDee.Rk4(quadtank, Ts, supersample=2)

nx = 4
ny = 2
nu = 2
p_true = [1.6, 1.6, 4.9, 0.2]

Tperiod = 200
t = 0:Ts:1000
u1 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2)) .+ 0.25)
u2 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2 .+ pi/2)) .+ 0.25)
u  = vcat.(u1,u2)
u = [u; 0 .* u[1:100]]
x0 = [2.5, 1.5, 3.2, 2.8]
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u, p_true)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.01randn(ny) for y in y]

isinteractive() && plot(
    plot(reduce(hcat, x)', title="States"),
    plot(reduce(hcat, u)', title="Inputs")
)

R1 = Diagonal([0.1, 0.1, 0.1, 0.1])
R2 = Diagonal((1e-2)^2 * ones(ny))


Y = reduce(hcat, y)
U = reduce(hcat, u)

d = iddata(Y, U, Ts)
isinteractive() && plot(d)




x00 = 0.5*[2.5, 1.5, 3, 3]
p0 = [1.4, 1.4, 5.1, 0.25]


@test_throws "Initial" ControlSystemIdentification.nonlinear_pem(d, discrete_dynamics, measurement, p0, Inf*x00, R1, 100R2, nu)

model = ControlSystemIdentification.nonlinear_pem(d, discrete_dynamics, measurement, p0, x00, R1, 100R2, nu)

p_opt = model.p

Σ = inv(model.Λ())


using LinearAlgebra
e0 = norm(p_true - p0) / norm(p_true)
eopt = norm(p_true - p_opt) / norm(p_true)
@test eopt < e0

e0 = norm(x0 - x00) / norm(x0)
eopt = norm(x0 - model.x0) / norm(x0)
@test eopt < e0



if isinteractive()
    scatter(p_opt, yerror=2sqrt.(diag(Σ[1:4, 1:4])), lab="Estimate")
    scatter!(p_true, lab="True")
    scatter!(p0, lab="Initial guess")
end

ysim = simulate(model, U)
ypred = predict(model, d, model.x0)


predplot(model, d)
simplot(model, d)



model2 = ControlSystemIdentification.nonlinear_pem(d, model; R2 = 1000R2, optimize_x0 = false)
@test model2.res.ssr <= model.res.ssr
@test model2.x0 == model.x0

display(model2) # For coverage purposes

## Regularization
γ = 1e4 # Strong regularization to more or less force the optimum to be at the initial guess
modelγ = ControlSystemIdentification.nonlinear_pem(d, discrete_dynamics, measurement, p0, x00, R1, 100R2, nu; γ, iterations=300, show_every=50, optimizer=Dogleg())
@test norm(modelγ.p - p0) < 0.01norm(model.p - p0)

γ = 1e4*ones(8) # Strong regularization to more or less force the optimum to be at the initial guess
modelγ = ControlSystemIdentification.nonlinear_pem(d, discrete_dynamics, measurement, p0, x00, R1, 100R2, nu; γ, iterations=300, show_every=50, optimizer=Dogleg())
@test norm(modelγ.p - p0) < 0.01norm(model.p - p0)


## Covariance scaling vs bootstrap (ny=1 and ny=3)

@testset "nonlinear_pem covariance scaling" begin
    @info "Testing nonlinear_pem covariance scaling against bootstrap (ny=1 and ny=3)"

    Ts_b = 1.0
    nT_b = 200
    σ_b = 0.3
    N_reps = 30

    dyn_bs(x, u, p, t) = SA[p[1] * x[1] + p[2] * u[1]]
    meas_1(x, u, p, t) = SA[x[1]]
    meas_3(x, u, p, t) = SA[x[1], 0.5 * x[1], -0.3 * x[1]]

    p_true_bs = [0.9, 1.0]
    x0_bs = SA[1.0]

    Random.seed!(0)
    u_vec = [SA[0.5 * randn()] for _ in 1:nT_b]
    U_bs = reduce(hcat, u_vec)

    function rollout_manual(f, x0, uvec, p)
        x = x0
        traj = Vector{typeof(x0)}(undef, length(uvec))
        for k in eachindex(uvec)
            traj[k] = x
            x = f(x, uvec[k], p, 0)
        end
        traj
    end
    x_traj = rollout_manual(dyn_bs, x0_bs, u_vec, p_true_bs)

    function bootstrap_compare(meas, ny, seed_offset)
        y_clean_vec = meas.(x_traj, u_vec, Ref(p_true_bs), 0)
        Y_clean = reduce(hcat, y_clean_vec)

        R1_bs = Matrix(1e-6 * I, 1, 1)
        R2_bs = Matrix(σ_b^2 * I, ny, ny)

        p_ests   = zeros(2, N_reps)
        var_ests = zeros(2, N_reps)

        redirect_stdout(devnull) do
            for rep in 1:N_reps
                Random.seed!(seed_offset + rep)
                Y_noisy = Y_clean .+ σ_b .* randn(ny, nT_b)
                d_rep = iddata(Y_noisy, U_bs, Ts_b)
                model_b = ControlSystemIdentification.nonlinear_pem(
                    d_rep, dyn_bs, meas, p_true_bs, [x0_bs...], R1_bs, R2_bs, 1,
                )
                p_ests[:, rep]   = model_b.p
                var_ests[:, rep] = diag(inv(model_b.Λ()))[1:2]
            end
        end
        vec(mean(var_ests, dims=2)), vec(var(p_ests, dims=2))
    end

    # ny = 1
    mean_est_1, emp_var_1 = bootstrap_compare(meas_1, 1, 100)
    @test 0.6 < mean_est_1[1] / emp_var_1[1] < 1.7
    @test 0.6 < mean_est_1[2] / emp_var_1[2] < 1.7

    # ny = 3 — catches the (T·ny vs T) bug in the previous master formula
    mean_est_3, emp_var_3 = bootstrap_compare(meas_3, 3, 200)
    @test 0.6 < mean_est_3[1] / emp_var_3[1] < 1.7
    @test 0.6 < mean_est_3[2] / emp_var_3[2] < 1.7
end
