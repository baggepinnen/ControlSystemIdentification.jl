G1 =  tf([0.3,0, 1], [1, -0.5, 0], 1)
G1 =  tf([0.3, 1], [1, -0.5], 1)
u = randn(N)
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
na, nb = 1,1
Gest = arx(d, na, nb, direct = true, inputdelay = 1)
Gest ≈ G1

direct = true
yx, A = getARXregressor(y, u, na, nb, direct = direct, inputdelay = 1)

w = A\yx
a, b = params2poly(w, na ,nb, direct = direct, inputdelay = 1)
Gest = tf(b, a, 1)

Ad = [A u[2:end] u[2:end]]
yx
Ad\yx
