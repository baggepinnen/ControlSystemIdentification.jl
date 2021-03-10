
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
    Gest, Hest, res = gls(d, na, nb, nd, returnResidual = true, maxiter = 10, verbose = true, δmin = 1e-3)
    @test isapprox(Gest, G, atol = 10e-2)
    @test isapprox(Hest, 1/D, atol = 10e-2)
    @test var(res .- e[2:end]) < 10e-3
    
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
    Gest, Hest = gls(d, na, nb, nd, maxiter = 10, verbose = true, δmin = 1e-3)
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
    Gest, Hest, res = gls(d, na, nb, nd, returnResidual = true, maxiter = 10, verbose = true, δmin = 1e-3)
    @test isapprox(Gest, G, atol = 10e-3)
    @test isapprox(Hest, 1/tf([1, -0.49], [1], 1), atol = 10e-2)
    
    #### S10 #### prior knowledge neccessary for identification
    A = tf([1, -0.5], [1, 0], 1)
    B = tf([0, 1], [1, 0], 1)
    G = minreal(B / A)
    D = tf([1, 0.5], [1,0], 1)
    H = minreal(1 / (D * A))
    
    u = rand(Normal(0, 1), N)
    e = rand(Normal(0, 10), N)
    y = sim(G, u)
    v = sim(H, e)
    yv = y.+ v
    d = iddata(yv, u, 1)
    ############
    na, nb , nd = 1, 1, 1
    Gest, Hest, res = gls(d, na, nb, nd, H = 1/D, returnResidual = true, maxiter = 10, verbose = true, δmin = 1e-3)
    @test isapprox(Gest, G, atol = 10e-2)
    @test isapprox(Hest, 1/D, atol = 10e-2)

    ## TODO ##
    # * MISO
    # * inputdelay 
end


##### remove later #####
# N = 500 # Number of samples used for simulation by Söderström
# time = 1:N
# sim(G, u) = lsim(G, u, time)[1][:]

# #### S1 ####
# A = tf([1, -0.8], [1, 0], 1)
# B = tf([0, 1], [1, 0], 1)
# G = minreal(B / A)
# D = tf([1, 0.7], [1,0], 1)
# H = minreal(1 / (D * A))
# den(D)
# u = rand(Normal(0, 1), N)
# e = rand(Normal(0, 1), N)
# y = sim(G, u)
# v = sim(H, e)
# yv = y.+ v
# d = iddata(yv, u, 1)
# ###########
# na, nb , nd = 1, 1, 1
# Gest, Hest, res = gls(d, na, nb, nd, returnResidual = true, maxiter = 8, verbose = true, δmin = 1e-8)
# isapprox(Gest, G, atol = 10e-2)
# isapprox(Hest, 1/D, atol = 10e-2)
# arx(d, na, nb)

# var(res .- e[1:end-1])
# var(res)
# plot(res)
# #### S2 ####
# A = tf([1, -0.8], [1, 0], 1)
# B = tf([0, 1], [1, 0], 1)
# G = minreal(B / A)
# D = tf([1, 0.7], [1, 0], 1)
# H = minreal((D / A))

# u = rand(Normal(0, 1), N)
# e = rand(Normal(0, 0.1), N)
# y = sim(G, u)
# v = sim(H, e)
# yv = y.+ v
# d = iddata(yv, u, 1)
# ############
# na, nb , nd = 1, 1, 1
# Gest, Hest, res = gls(d, na, nb, nd, returnResidual = true, maxiter = 10, verbose = true, δmin = 1e-3)
# isapprox(Gest, G, atol = 10e-3)
# isapprox(Hest, 1/tf([1, -0.49], [1], 1), atol = 10e-2)

# #### S3 ####
# A = tf([1, -0.8], [1, 0], 1)
# B = tf([0, 1], [1, 0], 1)
# G = minreal(B / A)
# D = tf([1, -1, 0.2], [1, 0, 0], 1)
# H = minreal((D / A))

# u = rand(Normal(0, 1), N)
# e = rand(Normal(0, 0.1), N)
# y = sim(G, u)
# v = sim(H, e)
# yv = y.+ v
# d = iddata(yv, u, 1)
# ############
# na, nb , nd = 1, 1, 2
# Gest, Hest, res = gls(d, na, nb, nd, returnResidual = true, maxiter = 10, verbose = true, δmin = 1e-5)
# isapprox(Gest, G, atol = 10e-3)
# isapprox(Hest, 1/tf([1, -0.607, 0.352], [1,0,0], 1), atol = 10e-2)

# #### S4 ####
# A = tf([1, -1.5, 0.7], [1, 0, 0], 1)
# B = tf([0, 1, 0.5], [1, 0, 0], 1)
# G = minreal(B / A)
# D = tf([1, 0.7], [1, 0], 1)
# H = minreal((D / A))

# u = rand(Normal(0, 1), N)
# e = rand(Normal(0, 0.1), N)
# y = sim(G, u)
# v = sim(H, e)
# yv = y.+ v
# d = iddata(yv, u, 1)
# ############

# #### S5 ####
# A = tf([1, -0.8], [1, 0], 1)
# B = tf([0, 1], [1, 0], 1)
# G = minreal(B / A)
# D = tf([1, -0.2], [1, 0], 1)
# H = minreal((1 / (D*A)))
# u = rand(Normal(0, 1), N)
# e = rand(Normal(0, 10), N)
# y = sim(G, u)
# v = sim(H, e)
# yv = y.+ v
# d = iddata(yv, u, 1)
# ############
# na, nb , nd = 1, 1, 1
# Gest, Hest, res = gls(d, na, nb, nd, returnResidual = true, maxiter = 10, verbose = true, δmin = 1e-3)
# isapprox(Gest, G, atol = 10e-2)
# isapprox(Hest, 1/D, atol = 10e-2)

# H = 1/D
# v = sim(H, e)
# ar(iddata(v), 1)

# #### S10 ####
# A = tf([1, -0.5], [1, 0], 1)
# B = tf([0, 1], [1, 0], 1)
# G = minreal(B / A)
# D = tf([1, 0.5], [1], 1)
# H = minreal(1 / (D * A))

# u = rand(Normal(0, 1), N)
# e = rand(Normal(0, 10), N)
# y = sim(G, u)
# v = sim(H, e)
# yv = y.+ v
# d = iddata(yv, u, 1)
# ############
# na, nb , nd = 1, 1, 1
# Gest, Hest, res = gls(d, na, nb, nd, H = 1/D, returnResidual = true, maxiter = 10, verbose = true, δmin = 1e-5)
# isapprox(Gest, G, atol = 10e-2)
# isapprox(Hest, 1/D, atol = 10e-2)
# arx(d, na, nb)

# #### S12 ####
# A = tf([1, -0.7], [1, 0], 1)
# B = tf([0, 1], [1, 0], 1)
# G = minreal(B / A)
# D = tf([1, 0.9], [1], 1)
# H = minreal(1 / (D * A))

# u = rand(Normal(0, 1), N)
# e = rand(Normal(0, sqrt(1.2)), N)
# y = sim(G, u)
# v = sim(H, e)
# yv = y.+ v
# d = iddata(yv, u, 1)
# ############
# na, nb , nd = 1, 1, 1
# Gest, Hest, res = gls(d, na, nb, nd, returnResidual = true, maxiter = 10, verbose = true, δmin = 1e-3)
# isapprox(Gest, G, atol = 10e-2)
# isapprox(Hest, 1/D, atol = 10e-2)
