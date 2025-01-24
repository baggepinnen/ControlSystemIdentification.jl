using ControlSystemIdentification, ControlSystemsBase, Statistics, Test, LinearAlgebra

ny = 2
nu = 3
nx = 6
Ts = 0.1
N = 300

G = c2d(tf(1, [1, 1]), Ts) |> ss
G = [G G G; G G G]
u = randn(nu, N)
x0 = randn(nx)
y, t, x = lsim(G, u; x0)
d = iddata(y, u, Ts)
model = subspaceid(d, nx)

e = ControlSystemIdentification.prediction_error(model, d)
@test size(e) == size(d.y)

pd = ControlSystemIdentification.predictiondata(d)

pesys = ControlSystemIdentification.prediction_error_filter(model)
e2 = predict(pesys, pd, zeros(pesys.nx))
# e3 = lsim(pesys, pd, x0=x0).y

@test mean(abs2, e2 .- e) < 0.01
@test mean(abs2, (e2 .- e)[:, nx+1:end]) < 1e-5

# plot((e2 .- e)')

# plot(e', layout=2)
# plot!(e2')
# plot!(e3')

residualplot(model, d)

