Random.seed!(1)
##
T = 200
nx = 2
nu = 1
ny = 1
x0 = randn(nx)
σy = 0.5
sim(sys, u) = lsim(sys, u, 1:T)[1]
sys = tf(1, [1, 2 * 0.1, 0.1])
sysn = tf(σy, [1, 2 * 0.1, 0.3])

u  = randn(nu, T)
un = u + 0.1randn(size(u))
y  = sim(sys, u)
yn = y + sim(sysn, σy * randn(size(u)))
dd = iddata(yn, un, 1)

uv  = randn(nu, T)
yv  = sim(sys, uv)
ynv = yv + sim(sysn, σy * randn(size(uv)))
dv  = iddata(yv, uv, 1)
dnv = iddata(ynv, uv, 1)
##

res = [
    ControlSystemIdentification.newpem(dnv, nx)
    for nx in [1, 3, 4]
]

ω = exp10.(range(-2, stop = log10(pi), length = 150))
fig = plot(layout = 4, size = (1000, 600))
for i in eachindex(res)
    (sysh, opt) = res[i]
    ControlSystemIdentification.simplot!(
        sysh,
        dnv;
        subplot = 1,
        ploty = i == 1,
    )
    ControlSystemIdentification.predplot!(
        sysh,
        dnv;
        subplot = 2,
        ploty = i == 1,
    )
end
bodeplot!(
    getindex.(res, 1),
    ω,
    plotphase = false,
    subplot = 3,
    title = "Process",
    linewidth = 2 * [4 3 2 1],
)
bodeplot!(
    noise_model.(getindex.(res, 1)),
    ω,
    plotphase = false,
    subplot = 4,
    linewidth = 2 * [4 3 2 1],
)
bodeplot!(
    sys,
    ω,
    plotphase = false,
    subplot = 3,
    lab = "True",
    linecolor = :blue,
    l = :dash,
    legend = :bottomleft,
    title = "System model",
)
bodeplot!(
    ControlSystems.innovation_form(ss(sys), syse = ss(sysn), R2 = σy^2 * I),
    ω,
    plotphase = false,
    subplot = 4,
    lab = "True",
    linecolor = :blue,
    l = :dash,
    ylims = (0.1, 100),
    legend = :bottomleft,
    title = "Noise model",
)
display(fig)