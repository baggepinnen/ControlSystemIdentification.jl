using ControlSystemIdentification, ControlSystems

ny = 2
nu = 3
nx = 2
Ts = 0.1
N = 300

G = ssrand(ny, nu, nx; Ts)
u = randn(nu, N)
x0 = randn(nx)
y, t, x = lsim(G, u; x0)
d = iddata(y, u, Ts)
model = subspaceid(d, nx)

e = ControlSystemIdentification.prediction_error(model, d)
@test size(e) == size(d.y)

pd = ControlSystemIdentification.predictiondata(d)

pesys = ControlSystemIdentification.prediction_error_filter(model)
e2 = predict(pesys, pd)

@test mean(abs2, e2 .- e) < 0.01
@test mean(abs2, (e2 .- e)[:, 4:end]) < 1e-5

residualplot(model, d)

