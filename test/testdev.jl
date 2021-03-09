N = 50
t = 1:N
G1 =  tf([0.3,0, 1], [1, -0.5, 0], 1)
G1 =  tf([0.3, 1], [1, -0.5], 1)
u = randn(N)
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
na, nb = 1,3
Gest = arx(d, na, nb, inputdelay = 0)
Gest ≈ G1

direct = true
na, nb = 1, 2
yx, A = getARXregressor(y, u, na, nb, inputdelay = 0)

w = A\yx
a, b = params2poly2(w, na ,nb, inputdelay = 0)
Gest = tf(b, a, 1)
Gest ≈ G1

Ad = [A u[2:end] u[2:end]]
yx
Ad\yx

na, nb, inputdelay = 1, 1, 2
G1 =  tf([0,1], [1, -0.5,0], 1)
u = randn(N)
y = lsim(G1, u, t)[1][:]
d = iddata(y, u, 1)
yx, A = getARXregressor(y, u, na, nb, inputdelay = inputdelay)
w = A\yx
a, b = params2poly2(w, na ,nb, inputdelay = 2)
Gest = minreal(tf(b, a, 1))
Gest ≈ G1