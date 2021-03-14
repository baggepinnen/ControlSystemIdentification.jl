

"""
    G, H = gls(d::InputOutputData, na::Int, nb::Union{Int, Vector{Int}}, nd::Int)

Estimate the model `Ay = Bu + He`, where `H = 1/D`, using generalized least-squares.  

# Arguments:
- `d`: iddata
- `na`: order of A
- `nb`: order of B, takes the form nb = [nb₁, nb₂...] in MISO estimation
- `nd`: order of D

# Keyword Arguments:
- `H`: prior knowledge about the noise model
- `inputdelay`: optinal delay of input, inputdelay = 0 results in a direct term, takes the form inputdelay = [d₁, d₂...] in MISO estimation 
- `λ`: reg param
- `estimator`: e.g. `\\,tls,irls,rtls`, the latter three require `using TotalLeastSquares`.
- `δmin`: Minimal change in the power of v, that specifies convergence.
- `maxiter`: maximum number of iterations.
- `verbose`: if true, more informmation is printed

"""
function gls(d::InputOutputData, na::Int, nb::Union{Int, Vector{Int}}, nd::Int; H::Union{TransferFunction, Nothing} = nothing, inputdelay = ones(Int, size(nb)), δmin::Real = 0.01, maxiter::Int = 10, returnResidual::Bool = false, estimator = \, verbose::Bool = false, λ::Real = 0)
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
        
        # -> What is a good way to filter input and output by D here? This is the point where I came to think about the ar process shouldnt lsim(1/H, y/u, t) work? 
        # Filter input/output according to errormodel H, after initialization
        if iter > 0
            yF = filt_D(H, output(d)')
            if ninputs(d) == 1
                uF = filt_D(H, input(d)')
            else
                uF = fill(1.0, size(d.u)) 
                for i in 1:ninputs(d)
                    uF[i,:] = filt_D(H, d.u[i,:])
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
		# yest = lsim(GF, input(d)', timeVec)[1][:]
        # v = output(d)' .- yest # This is different # -> this worked in my head but it does not
        v = residuals(GF, d)

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

    # Return AR residual if needed
    if returnResidual
        residual = getResidual(H, v)
        return GF, H, residual
    else
        return GF, H
    end
end

# Contruct the residual from an AR-process 
function getResidual(ar::TransferFunction, y)
    pred = predict(ar, y)
    n = length(y) - length(pred)
    residual = y[1+n:end] .- pred
    return residual
  end

# -> calculates the residual e = Ay - Bu, there must be a smarter way for this
function residuals(Gest, d)
    denominator = den(Gest[1, 1])[1]
    A = impRes2tf(denominator)
    res = lsim(A, output(d)', 1:length(d))[1][:]
    for i in 1:ninputs(d)
        numinator = num(Gest[1, i])[1]
        pad = zeros(max(0, (length(denominator) - length(numinator))))
        B = impRes2tf([pad; numinator])
        res -= lsim(B, input(d)[i,:], 1:length(d))[1][:]
    end   
    return res 
end

# impulse response to tf
function impRes2tf(h::Vector{<:Real})
    den = zeros(length(h))
    den[1] = 1
    return tf(h, den, 1)
end

# function filt_D(H::TransferFunction, sig)
#     a = den(H)[1]
#     b = zeros(length(a))
#     b[1] = 1
#     # b = [1]
#     res = filt(b, a, sig)
#     return res
# end

function filt_D(H::TransferFunction, sig)
    D = impRes2tf(den(H)[1])
    result = lsim(D, sig, 1:length(sig))[1][:]
    return result    
end