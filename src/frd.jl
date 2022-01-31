import ControlSystems: feedback

"""
    FRD(w,r)

Represents frequency-response data. `w` holds the frequency vector and `r` the response. Methods defined on this type include
- `+-*`
- `length, vec, sqrt`
- `plot`
- [`feedback`](@ref)
- [`freqvec`](@ref)
- [`tfest`](@ref) to estimate a rational model
- Indexing in the frequency domain using, e.g., `G[1Hz : 5Hz]`, `G[1rad : 5rad]`

If `r` represents a MIMO frequency response, the dimensions are `ny × nu × nω`. `freqresp` returns a `PermutedDimsArray` whose `.parent` field follows this convention.
"""
struct FRD{WT<:AbstractVector,RT<:AbstractArray} <: LTISystem{Continuous}
    w::WT
    r::RT
end
"Represents frequencies in Herz for indexing of `FRD` objects: frd[2Hz:10Hz]"
struct Hz <: Number
    i::Any
end
Base.:*(i, ::Type{Hz}) = Hz(i)

"Represents frequencies in rad/s for indexing of `FRD` objects: frd[2rad:10rad]"
struct rad <: Number
    i::Any
end
Base.:*(i, ::Type{rad}) = rad(i)
(::Colon)(start::Union{Hz,rad}, stop::Union{Hz,rad}) = (start, stop)

import Base: +, -, *, /, length, sqrt, getindex
FRD(w, s::LTISystem) = FRD(w, freqresp(s, w)[:, 1, 1])
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
Base.size(f::FRD) = (1, 1) # Size in the ControlSystems sense
Base.lastindex(f::FRD) = length(f)
function Base.getproperty(f::FRD, s::Symbol)
    s === :Ts && return 1 / ((f.w[2] - f.w[2]) / (2π))
    getfield(f, s)
end
Base.propertynames(f::FRD, private::Bool = false) = (fieldnames(typeof(f))..., :Ts)
ControlSystems.noutputs(f::FRD) = 1
ControlSystems.ninputs(f::FRD) = 1

sqrt(f::FRD) = FRD(f.w, sqrt.(f.r))
getindex(f::FRD, i) = FRD(f.w[i], f.r[i])
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

feedback(P::FRD, K::LTISystem) = feedback(P, freqresp(K, P.w)[:, 1, 1])
feedback(P::LTISystem, K::FRD) = feedback(freqresp(P, K.w)[:, 1, 1], K)


freqvec(h, k) = LinRange(0, π / h, length(k))

"""
    c2d(w::AbstractVector{<:Real}, Ts; w_prewarp = 0)
    c2d(frd::FRD, Ts; w_prewarp = 0)

Transform continuous-time frequency vector `w` or frequency-response data `frd` from continuous to discrete time using a bilinear (Tustin) transform. This is useful in cases where a frequency response is obtained through frequency-response analysis, and the function [`subspaceidf`](@ref) is to be used.
"""
function ControlSystems.c2d(w::AbstractVector{<:Real}, Ts; w_prewarp=0)
    a = w_prewarp == 0 ? Ts/2 : tan(w_prewarp*Ts/2)/w_prewarp
    @. 2*atan(w*a)
end

ControlSystems.c2d(f::FRD, Ts::Real; kwargs...) = FRD(c2d(f.w, Ts; kwargs...), f.r)


"""
    H, N = tfest(data, σ = 0.05)

Estimate a transfer function model using the Correlogram approach.
    Both `H` and `N` are of type `FRD` (frequency-response data).

`σ` determines the width of the Gaussian window applied to the estimated correlation functions before FFT. A larger `σ` implies less smoothing.
- `H` = Syu/Suu             Process transfer function
- `N` = Sy - |Syu|²/Suu     Noise PSD
"""
function tfest(d, σ::Real = 0.05)
    if d.ny > 1
        return [tfest(d[i,:], σ) for i in 1:d.ny]
    end
    y, u, h = time1(output(d)), time1(input(d)), sampletime(d)
    Syy, Suu, Syu = fft_corr(y, u, σ)
    w = freqvec(h, Syu)
    H = FRD(w, Syu ./ Suu)
    N = FRD(w, @.(Syy - abs2(Syu) / Suu) ./ length(y))
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
    κ = coherence(d; n = length(d)÷10, noverlap = n÷2, window=hamming)

Calculates the magnitude-squared coherence Function. κ close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy.
κ: Coherence function (not squared)
N: Noise model
"""
function coherence(d; n = length(d) ÷ 10, noverlap = n ÷ 2, window = hamming)
    noutputs(d) == 1 || throw(ArgumentError("coherence only supports a single output. Index the data object like `d[i,j]` to obtain the `i`:th output and the `j`:th input."))
    ninputs(d) == 1 || throw(ArgumentError("coherence only supports a single input. Index the data object like `d[i,j]` to obtain the `i`:th output and the `j`:th input."))
    y, u, h = time1(output(d)), time1(input(d)), sampletime(d)
    Syy, Suu, Syu = wcfft(y, u, n = n, noverlap = noverlap, window = window)
    k = (abs2.(Syu) ./ (Suu .* Syy))#[end÷2+1:end]
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
        Syu .+= xy .* conj.(xu)
        Syy .+= abs2.(xy)
        Suu .+= abs2.(xu)
    end
    Syy, Suu, Syu
end



"""
    ir, t, Σ = impulseest(d::AbstractIdData, n; λ=0, estimator=ls)

Estimates the system impulse response by fitting an `n`:th order FIR model. Returns impulse-response estimate, time vector and covariance matrix.
See also `impulseestplot`
"""
function impulseest(d::AbstractIdData, n; λ = 0, estimator = ls)
    h, y, u = d.Ts, time1(output(d)), time1(input(d))
    N = min(length(u), length(y))
    @views yt, A = getARXregressor(y[1:N], u[1:N], 0, n)
    ir = estimator(A, yt, λ)
    t = range(h, length = n, step = h)
    Σ = parameter_covariance(yt, A, ir, λ)
    ir ./ h, t, Σ
end
