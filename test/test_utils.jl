using Test
using ControlSystemIdentification
using ControlSystemIdentification: time1, time2, rms, sse, mse, fpe, modelfit
using Statistics
v = zeros(2)
M = zeros(2, 4)
vv = fill(v, 4)
@test time1(v) == v
@test time1(M) == M'
@test time1(vv) == M'
@test time2(v) == v'
@test time2(M) == M
@test time2(vv) == M

v = zeros(2)
M = zeros(1, 2)
vv = [[0.0], [0.0]]
for x in (v, M, vv), t in (M, vv)
    # @show typeof(t), typeof(x)
    @test oftype(t, t) == t
    @test typeof(oftype(t, x)) == typeof(t)
end

y = randn(1, 3)
@test modelfit(y, y) == [100]

@test modelfit(y[:], y[:]) == 100

@test fpe(zeros(5), 3) == 0
@test fpe(zeros(2, 5), 3) == 0

@test sse(y)[] ≈ sum(abs2, y)
@test mse(y)[] ≈ mean(abs2, y)

@test oftype(1f0, [2]) == 2f0

@test_throws ArgumentError oftype(randn(2), zeros(2, 2))
@test oftype(randn(2), [zeros(1) for i in 1:2]) == zeros(2)


@test time1(y) == [vec(y);;]
