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

@testset "detrend and prefilter" begin
    T = 10_000
    fs = 50.0
    ts = 1/fs
    f1 = 0.001  # low frequency
    f2 = 1.0    # high frequency
    t = (0:T-1) * ts

    s1 = @. sinpi(f1 * t)
    s2 = @. sinpi(f2 * t)
    u = @. s1 + s2
    y = @. s1 + s2 + 10
    d = iddata(y, u, ts)

    d_ = ControlSystemIdentification.detrend(d)
    @test mean(d_.y) ≈ 0 atol = 1e-10
    @test mean(d_.u) ≈ 0 atol = 1e-10

    d_low = ControlSystemIdentification.prefilter(d, 0, f1 * 10) # lowpass, should remove s2
    d_high = ControlSystemIdentification.prefilter(d, f1*10, Inf) # highpass, should remove s1
    d_band = ControlSystemIdentification.prefilter(d, f1*10, f2/10) # bandpass, should remove s1 and s2

    
    k = 2750 # transient time
    err_high = abs.((vec(d_high.u) - s2)[k:end-k])
    err_low = abs.((vec(d_low.u) - s1)[k:end-k])
    err_band = abs.((vec(d_band.u))[k:end-k])

    @test maximum(err_high) < 0.05
    @test maximum(err_low) < 0.01
    @test maximum(err_band) < 0.05
end