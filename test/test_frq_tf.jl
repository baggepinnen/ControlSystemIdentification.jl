using Optim, ControlSystemsBase, ControlSystemIdentification


## Generate fake data
w = exp10.(LinRange(-2, 2, 500))
Gtest = tf([1.2], [1.1, 0.0009, 1.2])
data = FRD(w, Gtest)

p0 = (a = [1.0, 1.0, 1.0], b = [1.1]) # Initial guess

Gest = tfest(data, p0, freq_weight = 0) |> minimum_phase
@test norm(poles(Gtest) - poles(Gest)) < 0.01
@test dcgain(Gtest) ≈ dcgain(Gest)

Gest = tfest(data, tf(p0.b, p0.a), freq_weight = 0) |> minimum_phase
@test norm(poles(Gtest) - poles(Gest)) < 0.01
@test norm(poles(Gtest) - poles(minimum_phase(Gest))) < 0.01
@test dcgain(Gtest) ≈ dcgain(Gest)

if isinteractive()
    bodeplot(Gtest, w)
    bodeplot!(Gest, w)
    bodeplot!(minimum_phase(Gest), w, l=(:dash,))
end


## Test fitting with basis functions
a = exp10.(LinRange(-1, 1, 7))
pols = ωζ2complex.(a, 0.1)
pols = [pols; conj.(pols)]
# pols = a
# basis = kautz(pols, 1/(200))
basis = laguerre_oo(1, 50)

@test count(1:4) do _
    Gest,p = tfest(data, basis)
    r_est = FRD(w, Gest)
    mean(abs2, log.(abs.(r_est.r)) .- log.(abs.(data.r))) < 0.05
end >= 2



if isinteractive()
    Gr, Gram = baltrunc(ss(Gest))
    Gmp = ControlSystemIdentification.minimum_phase(Gr)
    Gr = minreal(Gmp, rtol=1e-6)
    bodeplot(Gtest, w, show=false, lab="Gtest")
    bodeplot!(Gest, w, show=false, lab="Gest")
    bodeplot!(Gr, w, show=false, lab="Gr")
    bodeplot!(Gmp, w, show=true, lab="Gmp")
    display(current())
end


##
if isinteractive()
    t = 0:0.01:100
    u = randn(1,length(t))
    y = lsim(Gtest, u, t)
    yest = lsim(Gest, u, t)
    ymp = lsim(Gmp, u, t)
    plot([y, yest, ymp])
end


##
# bands = w
# desired = data.r[1:2:end]
# fi = remez(100, bands, abs.(desired), Hz=200, filter_type=filter_type_hilbert)

## FIR design

# plotly(show=false)

# function DSP.freqresp(filter::AbstractVector, w::AbstractVector, h::Real)
#     n = h/length(filter)
#     map(w) do w
#         sum(filter[i]*cis(-w*(i-1)*n) for i in eachindex(filter))
#     end
# end

# function firfit_generic_phase(data::FRD, M, Ts, λ=sqrt(eps()))
#     w = data.w .* Ts
#     maximum(w) ≤ π || throw(ArgumentError("Maximum frequency higher than nyquist"))
#     dvec(w, M) = [cis(-k*w) for k in 0:M-1]
#     Av = [dvec(w, M) for w in w]
#     A = reduce(hcat, Av)'
#     A[:, 1] .= 1
#     A = [A; conj.(A)]
#     Y = [data.r; conj.(data.r)]
#     b = A\Y
#     @show cond(A)
#     @show norm(imag(b))
#     b = real(b)
#     # [reverse(b[2:end]); b]
#     b
# end

# function firfit_linear_phase(data::FRD, M, Ts, λ=sqrt(eps()))
#     w = data.w .* Ts
#     maximum(w) ≤ π || throw(ArgumentError("Maximum frequency higher than nyquist"))
#     # dvec(w, M) = [cis(-k*w) for k in 0:M]
#     dvec(w, M) = [2cos(k*w) for k in 0:(M-1)÷2]
#     Av = [dvec(w, M) for w in w]
#     A = reduce(hcat, Av)'
#     A[:, 1] .= 1
#     A = [A; λ*I]
#     # A = [A; conj.(A)]
#     # Y = [data.r; conj.(data.r)]
#     Y = data.r
#     b = A\[abs.(Y); zeros((M-1)÷2+1)]
#     @show cond(A)
#     b = real(b)
#     [reverse(b[2:end]); b]
# end



# A = map(1:50) do l
#     wl = l*pi/51
#     (1+cos(4wl))/2
# end

# phi = map(1:50) do l
#     wl = l*pi/51
#     -50*wl*(1+0.2*sin(wl))
# end

# w = map(1:50) do l
#     wl = l*pi/51
# end

# data = FRD(w, A .* cis.(phi))
# ##
# w = 2pi .* exp10.(LinRange(-2, 2, 100))
# # w = range(0, pi, length=500)
# # w = 0:0.01:pi
# Gtest = tf([1.2], [1.1, 0.01, 1.2])
# data = FRD(w, Gtest)
# ##
# using PyCall
# sps = pyimport("scipy.signal")
# M = 101
# bs = sps.fir_filter_design.firls(M, data.w, abs.(data.r), fs=1000)

# ##

# # b = firfit_linear_phase(data, M, 0.001, 1)
# b = firfit_generic_phase(data, M, 0.001)
# # bodeplot(tf(b, [1], 1), w, xscale=:identity, yscale=:identity, ticks=:default)
# # bodeplot!(Gtest, w)

# plot([b bs])
# display(current())

# ##
# bodeplot(tf(b, [1], 1), w)
# bodeplot!(tf((bs), [1], 1), w, xscale=:log10, yscale=:log10, ticks=:default)
# display(current())


# ##
# bs = sps.fir_filter_design.firls(101, data.w, abs.(data.r), fs=2pi)
# Gfir = tf((bs), [1], 1)
# bodeplot(Gfir, w, ticks=:default)
# display(current())


# Generate test case for minreal
basis = laguerre_oo(1, 30)
@test count(1:4) do _
    Gest,p = tfest(data, basis)
    r_est = FRD(w, Gest)
    mean(abs2, log.(abs.(r_est.r)) .- log.(abs.(data.r))) < 0.05
end >= 2
# Gr, Gram = baltrunc(Gest)
# Gr = minreal(Gest)
# hinfnorm(Gr-Gest)