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
@test ControlSystemIdentification.modelfit(y, y) == [100]