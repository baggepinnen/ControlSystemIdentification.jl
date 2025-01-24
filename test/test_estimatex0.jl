using ControlSystemIdentification, ControlSystemsBase, LinearAlgebra, Test
sys = let
    tempA = [-0.8856235560821042 -0.22921539224367568 0.012490968503801405 -0.0031582023117939775; -0.0650107681965465 0.265109249479631 -0.11490729481821864 -0.0928635498943703; -0.07553654250010958 -0.3970489858403744 -0.12058107819259327 -0.047027277340539866; -0.020371527731964427 0.05621299034634304 0.043834656104102254 0.04811251441269695]
    tempB = [0.7346548163287362 0.8705727872717879 0.9632600257078154; -1.2652676231505515 1.0248285420827408 -0.4251583599975983; -0.13267693221045443 -0.22695212872197443 -0.755383722523026; -0.03507359451538111 -0.11258043034104165 -0.0380270790583697]
    tempC = [-0.5174456986544714 0.6206939128894883 0.9896748531545921 -0.021348328735838855; -1.449634395825332 1.3178101347439164 -0.4141647868572434 0.044910272305538325]
    tempD = 0*[1.588185809048972 0.854939942572876 0.0783542725724059; -1.076733535599696 0.9615389984422018 1.289421721394988]
    ss(tempA, tempB, tempC, tempD, 1)
end

R1 = I(sys.nx)
R2 = I(sys.ny)
K = kalman(sys, R1, R2)
sys2 = ControlSystemIdentification.PredictionStateSpace(sys, K, R1, R2)

##
using Random
Random.seed!(0)
# NOTE: These two were previously promoted to standard StateSpace, but are not any longer. We thus exposed a bug in estimate_x0 with PredictionStateSpace that has to be fixed
for sys in [sys, sys2]
    @show typeof(sys)
    local x0,x0h,u,y,t,x,d,yh,xh
    x0 = [0.1866890303969916, -2.10650892753435, -0.5132225023772066, -0.9640845058831279]
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

    x0h = estimate_x0(sys, d, 8, focus=:simulation)

    @test y[:,1] ≈ sys.C*x0h + sys.D*u[:,1]
    @test norm(x0-x0h)/norm(x0) < 1e-10

    x0h = ControlSystemIdentification.estimate_x0(sys, d, 8, fixed=[Inf, x0[2], Inf, Inf])
    @test x0h[2] == x0[2] # Should be exact equality

    @test y[:,1] ≈ sys.C*x0h + sys.D*u[:,1]
    @test norm(x0-x0h)/norm(x0) < 1e-10

    yh,_,xh = lsim(sys, u; x0=x0h)
    @test norm(y-yh)/norm(y) < 1e-10

end



# psys = prediction_error_filter(sys)
# pd = deepcopy(predictiondata(d))
# # pd.u[4:end, 1] .= 0
# x0p = estimate_x0(psys, pd, 50)
# ep = lsim(psys, pd; x0=x0p)
# plot(ep)