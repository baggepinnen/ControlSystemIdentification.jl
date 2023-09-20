import ControlSystemsBase: feedback

"""
    FRD(w, r)

Represents frequency-response data. `w` holds the frequency vector and `r` the response. Methods defined on this type include
- `+-*`
- `length, vec, sqrt`
- `plot`
- `feedback`
- [`freqvec`](@ref)
- [`tfest`](@ref) to estimate a rational model
- Indexing in the frequency domain using, e.g., `G[1Hz : 5Hz]`, `G[1rad : 5rad]`

If `r` represents a MIMO frequency response, the dimensions are `ny × nu × nω`.

An object `frd::FRD` can be plotted using `plot(frd, hz=false)` if `using Plots` has been called.
"""
struct FRD{WT<:AbstractVector,RT<:AbstractArray} <: LTISystem{Continuous}
    w::WT
    r::RT
end
"Represents frequencies in Herz for indexing of `FRD` objects: `frd[2Hz:10Hz]`"
struct Hz <: Number
    i::Any
end
Base.:*(i, ::Type{Hz}) = Hz(i)

"Represents frequencies in rad/s for indexing of `FRD` objects: `frd[2rad:10rad]`"
struct rad <: Number
    i::Any
end
Base.:*(i, ::Type{rad}) = rad(i)
(::Colon)(start::Union{Hz,rad}, stop::Union{Hz,rad}) = (start, stop)

import Base: +, -, *, /, length, sqrt, getindex

"""
    FRD(w, sys::LTISystem)

Generate a frequency-response data object by evaluating the frequency response of `sys` at frequencies `w`.
"""
function FRD(w, s::LTISystem)
    if ControlSystemsBase.issiso(s)
        FRD(w, freqrespv(s, w))
    else
        FRD(w, freqresp(s, w))
    end
end
Base.vec(f::FRD) = f.r
*(f::FRD, f2)    = FRD(f.w, f.r .* vec(f2))
+(f::FRD, f2)    = FRD(f.w, f.r .+ vec(f2))
-(f::FRD, f2)    = FRD(f.w, f.r .- vec(f2))
/(f::FRD, f2)    = FRD(f.w, f.r ./ vec(f2))

# Below are required for ambiguities
*(f::FRD, f2::FRD) = FRD(f.w, f.r .* f2.r)
+(f::FRD, f2::FRD) = FRD(f.w, f.r .+ f2.r)
-(f::FRD, f2::FRD) = FRD(f.w, f.r .- f2.r)
/(f::FRD, f2::FRD) = FRD(f.w, f.r ./ f2.r)

*(f::FRD, f2::LTISystem) = *(f, FRD(f.w, f2))
+(f::FRD, f2::LTISystem) = +(f, FRD(f.w, f2))
-(f::FRD, f2::LTISystem) = -(f, FRD(f.w, f2))

-(f::FRD) = FRD(f.w, -f.r)
length(f::FRD) = length(f.w)
Base.size(f::FRD) = (1, 1) # Size in the ControlSystemsBasesense
Base.lastindex(f::FRD) = length(f)
function Base.getproperty(f::FRD, s::Symbol)
    s === :Ts && return π / maximum(f.w)
    s === :nu && return ninputs(f)
    s === :ny && return noutputs(f)
    getfield(f, s)
end
Base.propertynames(f::FRD, private::Bool = false) = (fieldnames(typeof(f))..., :Ts)

function Base.show(io::IO, frd::FRD)
    write(io, "Frequency (rad/s)\n")
    write(io, "----------------\n")
    show(io, MIME("text/plain"), frd.w)
    write(io, "\n\n")
    write(io, "Response\n")
    write(io, "--------\n")
    show(io, MIME("text/plain"), frd.r)
end

ControlSystemsBase.noutputs(f::FRD) = f.r isa AbstractVector ? 1 : size(f.r, 1)
ControlSystemsBase.ninputs(f::FRD) = f.r isa AbstractVector ? 1 : size(f.r, 2)
ControlSystemsBase._default_freq_vector(f::Vector{<:FRD}, _) = f[1].w
function ControlSystemsBase.bode(f::FRD, w::AbstractVector = f.w; unwrap=true)
    w == f.w || error("Frequency vector must match the one stored in the FRD")
    angles = angle.(f.r)
    angles = reshape(angles, f.ny, f.nu, :)
    unwrap && ControlSystemsBase.unwrap!(angles,3)
    @. angles = rad2deg(angles)
    reshape(abs.(f.r), f.ny, f.nu, :), angles, f.w
end

function ControlSystemsBase.freqresp(f::FRD, w::AbstractVector{W} = f.w) where W <: Real
    w == f.w || error("Frequency vector must match the one stored in the FRD")
    reshape(f.r, f.ny, f.nu, :)
end

sqrt(f::FRD) = FRD(f.w, sqrt.(f.r))
function getindex(f::FRD, i)
    if ControlSystemsBase.issiso(f)
        FRD(f.w[i], f.r[i])
    else
        FRD(f.w[i], f.r[:,:,i])
    end
end
getindex(f::FRD, i::Int) = f.r[i]
getindex(f::FRD, i::Int, j::Int) = (@assert(i == 1 && j == 1); f)
function getindex(f::FRD, r::Tuple{Hz,Hz})
    s = findfirst(2pi * r[1].i .< f.w)
    e = findlast(f.w .< 2pi * r[2].i)
    f[s:e]
end
function getindex(f::FRD, r::Tuple{rad,rad})
    s = findfirst(r[1].i .< f.w)
    e = findlast(f.w .< r[2].i)
    f[s:e]
end
getindex(f::Tuple{<:FRD, <:FRD}, r::Tuple) = (f[1][r], f[2][r])
Base.isapprox(f1::FRD, f2::FRD; kwargs...) =
    (f1.w == f2.w) && isapprox(f1.r, f2.r; kwargs...)
Base.:(==)(f1::FRD, f2::FRD) = (f1.w == f2.w) && ==(f1.r, f2.r)

sensitivity(P::FRD, K) = FRD(P.w, 1.0 ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P::FRD, K) = FRD(P.w, vec(P) ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P::FRD, K::Number=1) = FRD(P.w, vec(P) ./ (1.0 .+ vec(P) .* K))
feedback(P::Number, K::FRD) = FRD(K.w, P ./ (1.0 .+ P .* vec(K)))
feedback(P, K::FRD) = FRD(K.w, vec(P) ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P::FRD, K::FRD) = feedback(P, vec(K))

feedback(P::FRD, K::LTISystem) = feedback(P, freqrespv(K, P.w))
feedback(P::LTISystem, K::FRD) = feedback(freqrespv(P, K.w), K)

"""
    freqvec(h, k)

Return a frequency vector of length `k` for systems with sample time `h`.
"""
freqvec(h, k::AbstractVector) = freqvec(h, length(k))
freqvec(h, k::Integer) = LinRange(0, π / h, k)

ControlSystemsBase.issiso(frd::FRD) = ndims(frd.r) == 1 || (size(frd.r, 1) == size(frd.r, 2) == 1)

"""
    c2d(w::AbstractVector{<:Real}, Ts; w_prewarp = 0)
    c2d(frd::FRD, Ts; w_prewarp = 0)

Transform continuous-time frequency vector `w` or frequency-response data `frd` from continuous to discrete time using a bilinear (Tustin) transform. This is useful in cases where a frequency response is obtained through frequency-response analysis, and the function [`subspaceid`](@ref) is to be used.
"""
function ControlSystemsBase.c2d(w::AbstractVector{<:Real}, Ts; w_prewarp=0)
    a = w_prewarp == 0 ? Ts/2 : tan(w_prewarp*Ts/2)/w_prewarp
    @. 2*atan(w*a)
end

ControlSystemsBase.c2d(f::FRD, Ts::Real; kwargs...) = FRD(c2d(f.w, Ts; kwargs...), f.r)


"""
    H, N = tfest(data, σ = 0.05, method = :corr)

Estimate a transfer function model using the Correlogram approach (default) using the signal model ``y = H(iω)u + n``.

Both `H` and `N` are of type `FRD` (frequency-response data).

- `σ` determines the width of the Gaussian window applied to the estimated correlation functions before FFT. A larger `σ` implies less smoothing.
- `H` = Syu/Suu             Process transfer function
- `N` = Sy - |Syu|²/Suu     Estimated Noise PSD (also an estimate of the variance of ``H``). Note that a PSD is related to the "noise model" ``N_m`` used in the system identification literature as ``N_{psd} = N_m^* N_m``. The magnitude curve of the noise model can be visualized by plotting `√(N)`.
- `method`: `:welch` or `:corr`. `:welch` uses the Welch method to estimate the power spectral density, while `:corr` (default) uses the Correlogram approach. If `method = :welch`, the additional keyword arguments `n`, `noverlap` and `window` determine the number of samples per segment (default 10% of data), the number of samples to overlap between segments (default 50%), and the window function to use (default `hamming`), respectively.

# Extended help
This estimation method is unbiased if the input ``u`` is uncorrelated with the noise ``n``, but is otherwise biased (e.g., for identification in closed loop).
"""
function tfest(d::AbstractIdData, σ::Real = 0.05; method = :corr, n = length(d) ÷ 10, noverlap = n ÷ 2, window = hamming)
    d.nu == 1 || error("Cannot perform tfest on multiple-input data. Consider using time-domain estimation or statespace estimation.")
    if d.ny > 1
        HNs = [tfest(d[i,1], σ) for i in 1:d.ny]
        HR = reshape(reduce(vcat, [transpose(hn[1].r) for hn in HNs]), d.ny, 1, :)
        NR = reshape(reduce(vcat, [transpose(hn[2].r) for hn in HNs]), d.ny, 1, :)
        return FRD(HNs[1][1].w, HR), FRD(HNs[1][1].w, NR)
    end
    y, u, h = vec(output(d)), vec(input(d)), sampletime(d)
    if method === :welch
        Syy, Suu, Syu = wcfft(y, u, n = n, noverlap = noverlap, window = window)
    elseif method === :corr
        Syy, Suu, Syu = fft_corr(y, u, σ)
    else error("Unknown method $method") end
    w = freqvec(h, Syu)
    H = FRD(w, Syu ./ Suu)
    N = FRD(w, real.(Syy .- abs2.(Syu) ./ Suu) ./ length(y))
    return H, N
end

function fft_corr(y, u, σ = 0.05)
    n = length(y)
    w = gaussian(2n - 1, σ)

    Syu = rfft(ifftshift(w .* xcorr(y, u)))
    Syy = rfft(ifftshift(w .* xcorr(y, y)))
    Suu = rfft(ifftshift(w .* xcorr(u, u)))
    Syy, Suu, Syu
end



"""
    κ² = coherence(d; n = length(d)÷10, noverlap = n÷2, window=hamming, method=:welch)

Calculates the magnitude-squared coherence Function. κ² close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ² << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy.

- κ: Squared coherence function in the form of an [`FRD`](@ref).
- `method`: `:welch` or `:corr`. `:welch` uses the Welch method to estimate the power spectral density, while `:corr` uses the Correlogram approach . For `method = :corr`, the additional keyword argument `σ` determines the width of the Gaussian window applied to the estimated correlation functions before FFT. A larger `σ` implies less smoothing.

See also [`coherenceplot`](@ref)

# Extended help:
For the signal model ``y = Gu + v``, ``κ²`` is defined as 
```math
κ(ω)^2 = \\dfrac{S_{uy}}{S_{uu} S_{yy}} = \\dfrac{|G(iω)|^2S_{uu}^2}{S_{uu} (|G(iω)|^2S_{uu}^2 + S_{vv})} = \\dfrac{1}{1 + \\dfrac{S_{vv}}{S_{uu}|G(iω)|^2}}
```
from which it is obvious that ``0 ≤ κ² ≤ 1`` and that κ² is close to 1 if the noise energy ``S_{vv}`` is small compared to the output energy due to the input ``S_{uu}|G(iω)|^2``.
"""
function coherence(d::AbstractIdData; n = length(d) ÷ 10, noverlap = n ÷ 2, window = hamming, method=:welch, σ = 0.05)
    noutputs(d) == 1 || throw(ArgumentError("coherence only supports a single output. Index the data object like `d[i,j]` to obtain the `i`:th output and the `j`:th input."))
    ninputs(d) == 1 || throw(ArgumentError("coherence only supports a single input. Index the data object like `d[i,j]` to obtain the `i`:th output and the `j`:th input."))
    y, u, h = vec(output(d)), vec(input(d)), sampletime(d)
    if method === :welch
        Syy, Suu, Syu = wcfft(y, u, n = n, noverlap = noverlap, window = window)
    else
        Syy0, Suu0, Syu = fft_corr(y, u, σ)
        Syy = real(Syy0)
        Suu = real(Suu0)
    end
    k = @. abs2(Syu) / (Suu * Syy) # [end÷2+1:end]
    Sch = FRD(freqvec(h, k), k)
    return Sch
end

function wcfft(y, u; n = length(y) ÷ 10, noverlap = n ÷ 2, window = hamming)
    win, norm2 = DSP.Periodograms.compute_window(window, n)
    uw = arraysplit(u, n, noverlap, nextfastfft(n), win)
    yw = arraysplit(y, n, noverlap, nextfastfft(n), win)
    Syy = zeros(length(uw[1]) ÷ 2 + 1)
    Suu = zeros(length(uw[1]) ÷ 2 + 1)
    Syu = zeros(ComplexF64, length(uw[1]) ÷ 2 + 1)
    for i in eachindex(uw)
        xy = rfft(yw[i])
        xu = rfft(uw[i])
        @. Syu += xy * conj(xu)
        @. Syy += abs2(xy)
        @. Suu += abs2(xu)
    end
    Syy, Suu, Syu
end



"""
    ir, t, Σ = impulseest(d::AbstractIdData, n; λ=0, estimator=ls)

Estimates the system impulse response by fitting an `n`:th order FIR model. Returns impulse-response estimate, time vector and covariance matrix.

This function only supports single-output data, use [`okid`](@ref) for multi-output data.

See also [`impulseestplot`](@ref) and [`okid`](@ref).
"""
function impulseest(d::AbstractIdData, n; λ = 0, estimator = ls)
    d.ny == 1 || error("impulseest only supports single-output data, consider using the function okid instead")
    h, y, u = d.Ts, time1(output(d)), time1(input(d))
    N = min(size(u, 1), size(y, 1))
    @views yt, A = getARXregressor(y[1:N], u[1:N,:], 0, d.nu == 1 ? n : fill(n, d.nu); inputdelay = d.nu == 1 ? 0 : zeros(Int, d.nu))
    A .*= h # We adjust for the sample time here in order to get both ir and Σ adjusted correctly
    ir = estimator(A, yt, λ)
    t = range(0, length = n, step = h)
    Σ = parameter_covariance(yt, A, ir, λ)
    if d.nu > 1
        ir = reshape(ir, :, d.nu)
    end
    ir, t, Σ
end
