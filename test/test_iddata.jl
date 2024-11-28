using ControlSystemIdentification: Sec
@info "Testing iddata"
@testset "vectors" begin
    T = 100
    y = randn(1, T)
    @show d = iddata(y)
    @test d isa ControlSystemIdentification.OutputData
    @test length(d) == T
    @test output(d) == y
    @test length(input(d)) == T
    @test !hasinput(d)
    @test ControlSystemIdentification.time2(y) == y
    @test sampletime(d) == 1
    @test d[1:10] isa typeof(d)
    @test length(d[1:10]) == 10
    @test length(timevec(d)) == length(d)

    @test_nowarn plot(d)

    u = randn(1, T)
    @show d = iddata(y, u)
    @test d isa ControlSystemIdentification.InputOutputData
    @test length(d) == T
    @test output(d) == y
    @test hasinput(d)
    @test input(d) == u
    @test ControlSystemIdentification.time2(y) == y
    @test sampletime(d) == 1
    @test d[1, 1] == d
    @test d[1:10] isa typeof(d)
    @test (d*10).u == 10u
    @test (10d).y == 10y

    @test length([d d]) == 2length(d)

    @test oftype(Matrix, output(d)) == y

    yr, A = getARXregressor(d, 2, 2)
    @test size(A, 2) == 4

    @test_nowarn plot(d)


    @testset "second indexing" begin
        @info "Testing second indexing"
        d = iddata(randn(1,10), randn(1,10), 0.5)
        @test d[1Sec:3Sec].y == d.y[1:1, 3:7]
    end

end

@testset "matrices" begin
    T = 10
    ny, nu = 2, 3
    y = randn(ny, T)
    @show d = iddata(y)
    @test d isa ControlSystemIdentification.OutputData
    @test length(d) == T
    @test output(d) == y
    @test !hasinput(d)
    @test ControlSystemIdentification.time1(y) == y'
    @test sampletime(d) == 1

    @test_nowarn plot(d)

    u = randn(nu, T)
    @show d = iddata(y, u)
    @test d isa ControlSystemIdentification.InputOutputData
    @test length(d) == T
    @test output(d) == y
    @test hasinput(d)
    @test input(d) == u
    @test sampletime(d) == 1
    @test (d*10).u == 10u
    @test (10d).y == 10y

    @test (d*10*I).u == 10*I*u
    @test (10*I*d).y == 10*I*y

    @test_nowarn plot(d)

    u = randn(T, nu)
    @show d = iddata(y, u, 2)
    @test d isa ControlSystemIdentification.InputOutputData
    @test length(d) == T
    @test output(d) == y
    @test hasinput(d)
    @test input(d) == u'
    @test ControlSystemIdentification.time1(input(d)) == u
    @test sampletime(d) == 2
    @test (d*10).u == 10u'
    @test (10d).y == 10y

    @test (d*10*I).u == (u*10*I)'
    @test (10*I*d).y == 10*I*y

    @test_nowarn plot(d)

end

@testset "vectors of vectors" begin
    T = 100
    ny, nu = 2, 3
    y = [randn(ny) for _ = 1:T]
    @show d = iddata(y)
    @test d isa ControlSystemIdentification.OutputData
    @test length(d) == T
    @test output(d) == y
    @test !hasinput(d)
    @test ControlSystemIdentification.time1(y) == reduce(hcat, y)'
    @test sampletime(d) == 1

    @test_nowarn plot(d)

    u = [randn(nu) for _ = 1:T]
    @show d = iddata(y, u)
    @test d isa ControlSystemIdentification.InputOutputData
    @test length(d) == T
    @test output(d) == y
    @test hasinput(d)
    @test input(d) == u
    @test sampletime(d) == 1

    u = randn(T, nu)
    @show d = iddata(y, u, 2)
    @test d isa ControlSystemIdentification.InputOutputData
    @test length(d) == T
    @test output(d) == y
    @test hasinput(d)
    @test input(d) == u'
    @test ControlSystemIdentification.time1(y) == reduce(hcat, y)'
    @test ControlSystemIdentification.time1(input(d)) == u
    @test sampletime(d) == 2

    @test_nowarn plot(d)

end


@testset "ramp in/out" begin
    T = 100
    u = randn(1, T)
    y = randn(1, T)
    d = iddata(y, u)
    d2 = ramp_in(d, 10)
    # plot(d); plot!(d2)
    for i = 0:9
        @test d2.y[i+1] == i/9*d.y[i+1]
    end

    d2 = ControlSystemIdentification.ramp_out(d, 10)
    # plot(d); plot!(d2)
    for i = 0:9
        @test d2.y[end-i] == i/9*d.y[end-i]
    end
end

@testset "indexing" begin
    T = 10
    u = randn(1, T)
    y = randn(1, T)
    d = iddata(y, u)
    @test d[2:4] == iddata(y[:, 2:4], u[:, 2:4])
end


@testset "resample" begin
    T = 100
    u = randn(1, T)
    y = randn(1, T)
    d = iddata(y, u, 0.1)
    d2 = ControlSystemIdentification.DSP.resample(d, 1/10)
    @test length(d2) == 10
    @test d2.y[:] == ControlSystemIdentification.DSP.resample(d.y[:], 1/10)
    @test d2.u[:] == ControlSystemIdentification.DSP.resample(d.u[:], 1/10)
end