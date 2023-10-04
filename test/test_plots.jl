using ControlSystemIdentification, ControlSystemsBase, Plots

G = c2d(tf(1, [1,1,1]), 0.02)
u = randn(1, 100)
res = lsim(G, u, range(0, length=100, step=0.02))
d = iddata(res)

crosscorplot(d)
autocorplot(d.y, d.Ts)


plot(d)

predplot(ss(G), d)
simplot(ss(G), d)
simplot(d, ss(G))

specplot(d)
welchplot(d)

Gh,_ = tfest(d)
plot(Gh, plotphase=true)

plot(tfest(d))

model, x0 = newpem(d, 2)

residualplot(model, d)
residualplot(model, d; x0)
residualplot(model, d; h=Inf)
residualplot(model, d; x0, h=Inf)
