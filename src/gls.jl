
import StatsBase.residuals

"""
    G, H, e = gls(d::InputOutputData, na::Int, nb::Union{Int, Vector{Int}}, nd::Int)

Estimate the model `Ay = Bu + v`, where `v = He` and `H = 1/D`, using generalized least-squares.  

# Arguments:
- `d`: iddata
- `na`: order of A
- `nb`: order of B, takes the form nb = [nb₁, nb₂...] in MISO estimation
- `nd`: order of D

# Keyword Arguments:
- `H`: prior knowledge about the noise model
- `inputdelay`: optinal delay of input, inputdelay = 0 results in a direct term, takes the form inputdelay = [d₁, d₂...] in MISO estimation 
- `λ`: `λ > 0` can be provided for L₂ regularization
- `estimator`: e.g. `\\,tls,irls,rtls`
- `δmin`: Minimal change in the power of v, that specifies convergence.
- `maxiter`: maximum number of iterations.
- `verbose`: if true, more informmation is printed

# Extended Help

# Example:
```jldoctest
julia> N = 500 
500

julia> sim(G, u) = lsim(G, u, 1:N)[1][:]
sim (generic function with 1 method)

julia> A = tf([1, -0.8], [1, 0], 1)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
1.0z - 0.8
----------
   1.0z

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> B = tf([0, 1], [1, 0], 1)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Int64}}
1
-
z

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> G = minreal(B / A)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
   1.0
----------
1.0z - 0.8

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> D = tf([1, 0.7], [1, 0], 1)
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
1.0z + 0.7
----------
   1.0z

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> H = 1 / D
TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
   1.0z
----------
1.0z + 0.7

Sample Time: 1 (seconds)
Discrete-time transfer function model

julia> u, e = randn(N), randn(N)
[...]

julia> y, v = sim(G, u), sim(H * (1/A), e) # simulate process
[...]

julia> d = iddata(y.+ v, u, 1)
InputOutput data of length 500 with 1 outputs and 1 inputs

julia> na, nb , nd = 1, 1, 1
(1, 1, 1)

julia> Gest, Hest, res = gls(d, na, nb, nd)
(G = TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
   0.9987917259291642
-------------------------
1.0z - 0.7937837464682017

Sample Time: 1 (seconds)
Discrete-time transfer function model, H = TransferFunction{Discrete{Int64}, ControlSystems.SisoRational{Float64}}
          1.0z
-------------------------
1.0z + 0.7019519225937721

Sample Time: 1 (seconds)
Discrete-time transfer function model, e = [...]
```
"""
function gls(d::InputOutputData, na::Int, nb::Union{Int, Vector{Int}}, nd::Int; H::Union{TransferFunction, Nothing} = nothing, inputdelay = ones(Int, size(nb)), δmin::Real = 0.01, maxiter::Int = 10, estimator = \, verbose::Bool = false, λ::Real = 0)
    # Input Checking
    na >= 0 || throw(ArgumentError("na($na) must be positive"))
    all(nb .>= 0 )|| throw(ArgumentError("nb($nb) must be positive"))
    nd >= 1 || throw(ArgumentError("nd($nd) must be positive"))
    maxiter >= 1 || throw(DomainError("maxiter($maxiter) must be >0"))
	δmin > 0 || throw(ArgumentError("δmin($δmin) must be positive"))
    ninputs(d) == length(nb) || throw(ArgumentError("Length of nb ($(length(nb))) must equal number of input signals ($(ninputs(d)))"))

    
    # 1. initialize H, GF and v to bring them to scope
    if H === nothing
        H = tf([0], [1], 1)
        iter = 0 # initialization
    else
        iter = 1 # initial noisemodel is known, H is present
    end
    GF = tf([0], [1], 1)
    v = 0

    # Iterate
    eOld = 0
    δ = δmin + 1
	timeVec = 1:length(d.y)
    while iter <= maxiter && δ >= δmin
        
        # Filter input/output according to errormodel H, after initialization
        if iter > 0
            yF = lsim(1/H, output(d)', timeVec)[1][:]
            if ninputs(d) == 1
                uF = lsim(1/H, input(d)', timeVec)[1][:]
            else
                uF = fill(1.0, size(d.u)) 
                for i in 1:ninputs(d)
                    uF[i, :] = lsim(1/H, d.u[i,:], timeVec)[1][:]
                end
            end
        else
            # pass unfiltered signal for initialization
            yF = copy(output(d))
            uF = copy(input(d))
        end
		dF = iddata(yF, uF, d.Ts)

		# 2. fit arx model
        GF = arx(dF, na, nb, estimator = estimator, λ = λ)

        # 3. Evaluate residuals
        v = residuals(GF, d)
        # yr, Ar = getARXregressor(d, na, nb)
        # w = params(GF)[1]
        # yest = Ar * w
        # v = yr .- yest


        # 4. check if converged
        e = var(v)
        if eOld != 0
            δ = abs((eOld - e) / e)
        end
        verbose && println("iter: $iter, δ: $δ, e: $e")
        eOld = e

        # 5. estimate new noise model from residuals
        dH = iddata(v, d.Ts)
        H = ar(dH, nd, estimator = estimator)
        
        iter += 1
    end

    # Return AR residuals
    e = lsim(1/H, v, 1:length(v))[1][:]
    return (G = GF, H = H, e = e)
end

# calculates the arx residuals v = Ay - Bu
function residuals(ARX::TransferFunction, d::InputOutputData)
    size(ARX, 2) == ninputs(d) || throw(DomainError(d, "number of inputs $(ninputs(d)) does not match ARX model (expects $(size(ARX, 2)) inputs)"))

    w, a, b, inputdelay = params(ARX)
    na = length(a)
    nb = map(length, vec(b))
    
    y, A = getARXregressor(d, na, nb, inputdelay = inputdelay)
    ypred = A * w
    v = y .- ypred
    return v
end

# predict, inspired, by the predict for ar
function predict(ARX::TransferFunction, d::InputOutputData)
    size(ARX, 2) == ninputs(d) || throw(DomainError(d, "number of inputs $(ninputs(d)) does not match ARX model (expects $(size(ARX, 2)) inputs)"))

    w, a, b, inputdelay = params(ARX)
    na = length(a)
    nb = map(length, vec(b))
    
    y, A = getARXregressor(d, na, nb, inputdelay = inputdelay)
    ypred = A * w
    return ypred
end

# # -> calculates the arx residuals v = Ay - Bu, there must be a smarter way for this
# function residuals(Gest::TransferFunction, d::InputOutputData)
#     denumerator = den(Gest[1, 1])[1]
#     A = impRes2tf(denumerator) # A complicated way to split up the tf G = B/A into A and B, to use lsim for filtering
#     res = lsim(A, output(d)', 1:length(d))[1][:]
#     for i in 1:ninputs(d)
#         numerator = num(Gest[1, i])[1]
#         pad = zeros(max(0, (length(denumerator) - length(numerator))))
#         B = impRes2tf([pad; numerator])
#         res -= lsim(B, input(d)[i,:], 1:length(d))[1][:]
#     end   
#     return res 
# end

# # impulse response to tf, helper that pads zeros to the denominator to make the tf proper
# function impRes2tf(h::Vector{<:Real})
#     den = zeros(length(h))
#     den[1] = 1
#     return tf(h, den, 1)
# end
