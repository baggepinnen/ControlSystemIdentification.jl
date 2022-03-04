T = 200
h = 0.1
t = h:h:T
sim(sys, u) = lsim(sys, u, t)[1]
sys = c2d(tf(1, [1, 2 * 0.1, 0.1]), h)

u = randn(1,length(t))
y = sim(sys, u) + 0.1randn(1,length(t))

d = iddata(y, u, h)
impulseestplot(d, Int(50 / h), λ = 0)
plot!(impulse(sys, 50), l = (:dash, :blue))