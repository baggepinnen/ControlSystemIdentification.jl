using Optim, ControlSystems, ControlSystemIdentification


## Generate fake data
w = exp10.(LinRange(-1, 2, 300))
Gtest = tf([1.2], [1.1, 0.0009, 1.2])
data = FRD(w, Gtest)

p0 = (a = [1.0, 1.0, 1.0], b = [1.1]) # Initial guess

Gest = tfest(data, p0, freq_weight = 0)
@test pole(Gtest) ≈ pole(Gest)
@test dcgain(Gtest) ≈ dcgain(Gest)

Gest = tfest(data, tf(p0.b, p0.a), freq_weight = 0)
@test pole(Gtest) ≈ pole(Gest)
@test dcgain(Gtest) ≈ dcgain(Gest)

if isinteractive()
    bodeplot(Gtest, w)
    bodeplot!(Gest, w)
end


##
a = exp10.(LinRange(-1, 1, 7))
poles = ωζ2complex.(a, 0.1)
poles = [poles; conj.(poles)]
# poles = a
# basis = kautz(poles, 1/(200))
basis = laguerre_oo(1, 55)

Gest,p = tfest(data, basis)
r_est = FRD(w, Gest)

@test mean(abs2, log.(abs.(r_est.r)) .- log.(abs.(data.r))) < 0.01

if isinteractive()
    bodeplot(Gtest, w, show=false)
    bodeplot!(Gest, w, show=true)
end
