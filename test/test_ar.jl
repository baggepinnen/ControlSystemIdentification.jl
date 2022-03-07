using ControlSystemIdentification: rms
N = 10000
t = 1:N
y = zeros(1,N)
y[1] = -0.2
u = copy(y)
for i = 2:N
    y[i] = 0.9y[i-1]
end
# G = tf(1, [1, -0.9], 1)
G = tf([1, 0], [1, -0.9], 1)
y2 = lsim(G, u, t)[1]
@test all(y .== y2)

na = 1
yr, A = getARregressor(y[:], na)
@test length(yr) == N - na
@test size(A) == (N - na, na)

@test yr == y[na+1:end]
@test A[:, 1] == y[1:end-na]

d = iddata(y, 1)
Gh = ar(d, na)
@test Gh ≈ G # We should be able to recover this transfer function
uh = lsim(1/Gh, y, t)[1]
@test u ≈ uh atol = eps()

N = 10000
t = 1:N
y = zeros(1,N)
y[1] = 5randn()
for i = 2:N
    y[i] = 0.9y[i-1] + 0.01randn()
end
d = iddata(y, 1)
Gh = ar(d, na)
@test Gh ≈ G atol = 0.02 # We should be able to recover this transfer function
@test freqresptest(G, Gh) < 0.075
yh = predict(Gh, y)
@test rms(y[1, 2:end] - yh[:]) < 0.0105

Gh2 = ar(d, na, stochastic = true)
@test denvec(Gh2)[1][end] ≈ denvec(Gh)[1][end]