# Vector Autoregressive with External inputs (VARX)
In some fields, the use of [VARX](https://en.wikipedia.org/wiki/Vector_autoregression) is widespread. VARX models are special cases of general linear models on the form
```math
\begin{aligned}
x^+ &= Ax + Bu + Ke\\
y &= Cx + Du + e
\end{aligned}
```
which we show here by simulating the VARX model
```math
y_t = A_1 y_{t-1} + A_2 y_{t-2} + B_1 u_{t-1}
```
and estimating a regular statespace model. 


```@example VARX
using ControlSystemIdentification, Plots

A1 = 0.3randn(2,2)
A2 = 0.3randn(2,2)
B1 = randn(2,2)

N = 300
Y = [randn(2), randn(2)]
u = randn(2, N)

for i = 3:N
    yi = A1*Y[i-1] + A2*Y[i-2] + B1*u[:,i-1]
    push!(Y, yi)
end

y = hcat(Y...)

d = iddata(y, u, 1)
```

We now estimate two models, one using subspace-based identification ([`subspaceid`](@ref)) and one using the prediction-error method ([`newpem`](@ref)). The VARX model we estimated had a state of order 4, two lags of ``y``, each of which is a vector of length 2, we thus estimate models of order 4 below.
```@example VARX
model1 = subspaceid(d, 4, zeroD=true, s1=2)
model2, _ = newpem(d, 4)

plot(
    simplot(d, model1, title="Simulation performance subspaceid", layout=2),
    simplot(d, model2, title="Simulation performance PEM", layout=2),
)
```
The simulation indicates that the fit is close to 100%, i.e., the general linear model fit the VARX model perfectly, it does not however have exactly the same structure as the original VARX model. 

The estimated models on statespace form can be converted to MIMO transfer functions (polynomial models) by calling `tf`:
```@example VARX
using ControlSystemsBase: tf
tf(model2.sys)
```

## Summary
The statespace model
```math
\begin{aligned}
x^+ &= Ax + Bu + Ke\\
y &= Cx + Du + e
\end{aligned}
```
is very general and subsumes a lot of other, special-structure models, like the VARX model.