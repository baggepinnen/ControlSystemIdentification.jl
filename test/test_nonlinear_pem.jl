using ControlSystemIdentification, SeeToDee, LowLevelParticleFilters


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
x0 = Float64[2,2,3,3]
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u, p_true)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.01randn(ny) for y in y]

plot(
    plot(reduce(hcat, x)', title="States"),
    plot(reduce(hcat, u)', title="Inputs")
)

R1 = Diagonal([0.1, 0.1, 0.1, 0.1])
R2 = Diagonal((1e-2)^2 * ones(ny))


Y = reduce(hcat, y)
U = reduce(hcat, u)

d = iddata(Y, U, Ts)
plot(d)




x0 = 0.5*[2, 2, 3, 3]
p0 = [1.4, 1.4, 5.1, 0.25]

result = ControlSystemIdentification.nonlinear_pem(d, discrete_dynamics, measurement, p0, x0, 10R1, R2, nu)

p_opt = result.p

Σ = inv(result.Λ())


using LinearAlgebra
norm(p_true - p0) / norm(p_true)
norm(p_true - p_opt) / norm(p_true)

scatter(p_opt, yerror=2sqrt.(diag(Σ[1:4, 1:4])), lab="Estimate")
scatter!(p_true, lab="True")
scatter!(p0, lab="Initial guess")