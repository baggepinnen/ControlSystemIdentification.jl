
@testset "gls" begin
    @info "Testing gls"
    # Test examples are taken from Söderstöms paper and compared against it (this is also where the (high) tolerances come from)
    N = 500 # Number of samples used for simulation by Söderström
    time = 1:N
    sim(G, u) = lsim(G, u, time)[1][:]
    
    #### S1 ####
    A = tf([1, -0.8], [1, 0], 1)
    B = tf([0, 1], [1, 0], 1)
    G = minreal(B / A)
    D = tf([1, 0.7], [1, 0], 1)
    H = minreal(1 / (D * A))
    
    u = rand(Normal(0, 1), N)
    e = rand(Normal(0, 1), N)
    y = sim(G, u)
    v = sim(H, e)
    yv = y.+ v
    d = iddata(yv, u, 1)
    ###########
    na, nb , nd = 1, 1, 1
    Gest, Hest, res = gls(d, na, nb, nd, maxiter = 10, verbose = true, δmin = 1e-3)
    @test isapprox(Gest, G, atol = 10e-2)
    @test isapprox(Hest, 1/D, atol = 10e-2)
    @test var(res .- e[2:end]) < 10e-2
    
    #### S12 ####
    A = tf([1, -0.7], [1, 0], 1)
    B = tf([0, 1], [1, 0], 1)
    G = minreal(B / A)
    D = tf([1, 0.9], [1, 0], 1)
    H = minreal(1 / (D * A))
    
    u = rand(Normal(0, 1), N)
    e = rand(Normal(0, sqrt(1.2)), N)
    y = sim(G, u)
    v = sim(H, e)
    yv = y.+ v
    d = iddata(yv, u, 1)
    ############
    na, nb , nd = 1, 1, 1
    Gest, Hest, res = gls(d, na, nb, nd, maxiter = 10, verbose = true, δmin = 1e-3)
    @test isapprox(Gest, G, atol = 10e-2)
    @test isapprox(Hest, 1/D, atol = 10e-2)
    
    #### S2 #### structure of Ay = Bu + Ce
    A = tf([1, -0.8], [1, 0], 1)
    B = tf([0, 1], [1, 0], 1)
    G = minreal(B / A)
    D = tf([1, 0.7], [1, 0], 1)
    H = minreal((D / A))
    
    u = rand(Normal(0, 1), N)
    e = rand(Normal(0, 0.1), N)
    y = sim(G, u)
    v = sim(H, e)
    yv = y.+ v
    d = iddata(yv, u, 1)
    ############
    na, nb , nd = 1, 1, 1
    Gest, Hest, res = gls(d, na, nb, nd, maxiter = 10, verbose = true, δmin = 1e-3)
    @test isapprox(Gest, G, atol = 10e-3)
    @test isapprox(Hest, 1/tf([1, -0.49], [1, 0], 1), atol = 10e-2)
    
    #### S10 #### prior knowledge neccessary for identification
    N = 2000
    A = tf([1, -0.5], [1, 0], 1)
    B = tf([0, 1], [1, 0], 1)
    G = minreal(B / A)
    D = tf([1, 0.5], [1,0], 1)
    H = minreal(1 / (D * A))
    
    u = rand(Normal(0, 1), N)
    e = rand(Normal(0, 5), N)
    y = lsim(G, u, 1:N)[1][:]
    v = lsim(H, e, 1:N)[1][:]
    yv = y.+ v
    d = iddata(yv, u, 1)
    ############
    N = 500
    na, nb , nd = 1, 1, 1
    Gest, Hest, res = gls(d, na, nb, nd, H = 1/D, maxiter = 10, verbose = true, δmin = 1e-3)
    @test isapprox(Gest, G, atol = 20e-2)
    @test isapprox(Hest, 1/D, atol = 10e-2)

    # MISO
    A = tf([1, -0.8], [1, 0], 1)
    B1 = tf([0, 1], [1, 0], 1)
    B2 = tf([0, -1], [1, 0], 1)
    G1 = minreal(B1 / A)
    G2 = minreal(B2 / A)
    G = [G1 G2]
    D = tf([1, 0.7], [1, 0], 1)
    H = minreal(1 / (D * A))
    
    u1 = rand(Normal(0, 1), N)
    u2 = rand(Normal(0, 1), N)
    u = [u1 u2]
    e = rand(Normal(0, 1), N)
    y = sim(G, u)
    v = sim(H, e)
    yv = y.+ v
    d = iddata(yv, u', 1)
    ###########
    na, nb , nd = 1, [1, 1], 1
    Gest, Hest, res = gls(d, na, nb, nd, maxiter = 10, verbose = true, δmin = 1e-3)
    @test isapprox(Gest, G, atol = 10e-2)
    @test isapprox(Hest, 1/D, atol = 10e-2)
    ## TODO ##
    # * inputdelay 
end

# # Example
# N = 500 
# sim(G, u) = lsim(G, u, 1:N)[1][:]
# A = tf([1, -0.8], [1, 0], 1)
# B = tf([0, 1], [1, 0], 1)
# G = minreal(B / A)
# D = tf([1, 0.7], [1, 0], 1)
# H = minreal(1 / D)

# u, e = randn(N), randn(N)
# y, v = sim(G, u), sim(H * (1/A), e)
# d = iddata(y.+ v, u, 1)
# na, nb , nd = 1, 1, 1
# Gest, Hest, res = gls(d, na, nb, nd)
