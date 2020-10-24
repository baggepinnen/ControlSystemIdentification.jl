using Optim, ControlSystems, ControlSystemIdentification


## Generate fake data
w = exp10.(LinRange(-1, 2, 300))
Gtest = tf([1.2], [1.1, 0.0009, 1.2])
data = FRD(w, Gtest)

p0 = (a = [1.0, 1.0, 1.0], b = [1.1]) # Initial guess

Gest = arma(data, p0, freq_weight=0)
@test pole(Gtest) ≈ pole(Gest)
@test dcgain(Gtest) ≈ dcgain(Gest)

Gest = arma(data, tf(p0.b, p0.a), freq_weight=0)
@test pole(Gtest) ≈ pole(Gest)
@test dcgain(Gtest) ≈ dcgain(Gest)

if isinteractive()
    bodeplot(Gtest, w)
    bodeplot!(Gest, w)
end
