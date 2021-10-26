using ControlSystemIdentification, ControlSystems
sys = ssrand(2,3,4, Ts=1)
sys = balreal(sys)[1]

R1 = I(sys.nx)
R2 = I(sys.ny)
K = kalman(sys, R1, R2)
sys2 = ControlSystemIdentification.PredictionStateSpace(sys, K, R1, R2)


for sys in [sys, sys2]
    local x0,x0h,u,y,t,x,d,yh,xh
    x0 = randn(4)
    u = randn(sys.nu, 100)
    y,t,x = lsim(sys, u; x0)

    @test x[:,1] == x0
    @test y[:,1] ≈ sys.C*x0 + sys.D*u[:,1]
    @test x[:,2] ≈ sys.A*x0 + sys.B*u[:,1]
    @test y[:,2] ≈ sys.C*x[:,2] + sys.D*u[:,2]


    d = iddata(y, u, 1)
    x0h = estimate_x0(sys, d, 8)

    @test y[:,1] ≈ sys.C*x0h + sys.D*u[:,1]
    @test norm(x0-x0h)/norm(x0) < 1e-10

    yh,_,xh = lsim(sys, u; x0=x0h)
    @test norm(y-yh)/norm(y) < 1e-10
end
