# Test examples are taken from Söderstöms paper and compared against it (this is also where the (high) tolerances come from)
Random.seed!(0)
N = 500 # Number of samples used for simulation by Söderström
time = 1:N
sim(G, u) = lsim(G, u, time)[1][:]

#### S1 ####
A = tf([1, -0.8], [1, 0], 1)
B = tf([0, 1], [1, 0], 1)
G = minreal(B / A)
D = tf([1, 0.7], [1, 0], 1)
H = minreal(1 / (D * A))

u = randn(1,N)
e = randn(1,N)
y = sim(G, u)
v = sim(H, e)
yv = y.+ v
d = iddata(yv, u, 1)
###########
na, nb , nd = 1, 1, 1
Gest, Hest, res = arxar(d, na, nb, nd, iterations = 10, verbose = true, δmin = 1e-3)
@test isapprox(Gest, G, atol = 1e-1)
@test isapprox(Hest, 1/D, atol = 1e-1)
@test var(res .- e[2:end]) < 1e-1

#### S12 ####
A = tf([1, -0.7], [1, 0], 1)
B = tf([0, 1], [1, 0], 1)
G = minreal(B / A)
D = tf([1, 0.9], [1, 0], 1)
H = minreal(1 / (D * A))

u = randn(1,N)
e = sqrt(1.2)*randn(1, N)
y = sim(G, u)
v = sim(H, e)
yv = y.+ v
d = iddata(yv, u, 1)
############
na, nb , nd = 1, 1, 1
Gest, Hest, res = arxar(d, na, nb, nd, iterations = 10, verbose = true, δmin = 1e-3)
@test isapprox(Gest, G, atol = 2e-1)
@test isapprox(Hest, 1/D, atol = 1e-1)

#### S2 #### structure of Ay = Bu + Ce
A = tf([1, -0.8], [1, 0], 1)
B = tf([0, 1], [1, 0], 1)
G = minreal(B / A)
D = tf([1, 0.7], [1, 0], 1)
H = minreal((D / A))

u = randn(1, N)
e = 0.1randn(1, N)
y = sim(G, u)
v = sim(H, e)
yv = y .+ v
d = iddata(yv, u, 1)
############
na, nb , nd = 1, 1, 1
Gest, Hest, res = arxar(d, na, nb, nd, iterations = 10, verbose = true, δmin = 1e-3)
@test isapprox(Gest, G, atol = 1e-1)
@test isapprox(Hest, 1/tf([1, -0.49], [1, 0], 1), atol = 1e-1)

#### S10 #### prior knowledge neccessary for identification
N = 2000
A = tf([1, -0.5], [1, 0], 1)
B = tf([0, 1], [1, 0], 1)
G = minreal(B / A)
D = tf([1, 0.5], [1,0], 1)
H = minreal(1 / (D * A))

u = randn(1, N)
e = 5randn(1, N)
y = lsim(G, u, 1:N)[1][:]
v = lsim(H, e, 1:N)[1][:]
yv = y.+ v
d = iddata(yv, u, 1)
############
na, nb , nd = 1, 1, 1
Gest, Hest, res = arxar(d, na, nb, nd, H = 1/D, iterations = 10, verbose = true, δmin = 1e-3)
@test isapprox(Gest, G, atol = 4e-1)
@test isapprox(Hest, 1/D, atol = 1e-1)
@test freqresptest(G, Gest) < 0.5

# MISO
N = 500
A = tf([1, -0.8], [1, 0], 1)
B1 = tf([0, 1], [1, 0], 1)
B2 = tf([0, -1], [1, 0], 1)
G1 = minreal(B1 / A)
G2 = minreal(B2 / A)
G = [G1 G2]
D = tf([1, 0.7], [1, 0], 1)
H = minreal(1 / (D * A))

u1 = randn(1, N)
u2 = randn(1, N)
u = [u1; u2]
e = randn(1, N)
y = sim(G, u)
v = sim(H, e)
yv = y.+ v
d = iddata(yv, u, 1)
###########
na, nb , nd = 1, [1, 1], 1
Gest, Hest, res = arxar(d, na, nb, nd, iterations = 10, verbose = true, δmin = 1e-3)
@test isapprox(Gest, G, atol = 3e-1)
@test isapprox(Hest, 1/D, atol = 3e-1)

# inputdelay 
A = tf([1, -0.8], [1, 0], 1)
B = tf([0, 1], [1, 0, 0], 1)
G = minreal(B / A)
D = tf([1, 0.7], [1, 0], 1)
H = minreal(1 / (D * A))

u = randn(1,N)
e = randn(1,N)
y = sim(G, u)
v = sim(H, e)
yv = y.+ v
d = iddata(yv, u, 1)
###########
na, nb , nd, inputdelay = 1, 1, 1, 2
Gest, Hest, res = arxar(d, na, nb, nd, inputdelay = inputdelay, iterations = 10, verbose = true, δmin = 1e-3)
@test isapprox(Gest, G, atol = 1e-1)
@test isapprox(Hest, 1/D, atol = 1e-1)
@test var(res .- e[3:end]) < 1e-1