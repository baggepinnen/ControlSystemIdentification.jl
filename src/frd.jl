import ControlSystems: feedback

"""
    FRD(w,r)

Represents frequency-response data. `w` holds the frequency vector and `r` the response. Methods defined on this type include
- `+-*`
- `length, vec, sqrt`
- `plot`
- `feedback`
"""
struct FRD{WT<:AbstractVector,RT<:AbstractVector} <: LTISystem
    w::WT
    r::RT
end

struct Hz <: Number
    i::Any
end
Base.:*(i, ::Type{Hz}) = Hz(i)
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
Base.isapprox(f1::FRD, f2::FRD; kwargs...) =
    (f1.w == f2.w) && isapprox(f1.r, f2.r; kwargs...)
Base.:(==)(f1::FRD, f2::FRD) = (f1.w == f2.w) && ==(f1.r, f2.r)

sensitivity(P::FRD, K) = FRD(P.w, 1.0 ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P::FRD, K) = FRD(P.w, vec(P) ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P, K::FRD) = FRD(K.w, vec(P) ./ (1.0 .+ vec(P) .* vec(K)))
feedback(P::FRD, K::FRD) = feedback(P, vec(K))

feedback(P::FRD, K::LTISystem) = feedback(P, freqresp(K, P.w)[:, 1, 1])
feedback(P::LTISystem, K::FRD) = feedback(freqresp(P, K.w)[:, 1, 1], K)

@recipe function plot_frd(frd::FRD; hz = false, plotphase=false)
    yscale --> :log10
    xscale --> :log10
    xguide --> (hz ? "Frequency [Hz]" : "Frequency [rad/s]")
    yguide --> "Magnitude"
    title --> "Bode Plot"
    legend --> false
    layout --> (plotphase ? 2 : 1)
    @series begin
        inds = findall(x -> x == 0, frd.w)
        subplot --> 1
        useinds = setdiff(1:length(frd.w), inds)
        (hz ? 1 / (2π) : 1) .* frd.w[useinds], abs.(frd.r[useinds])
    end
    if plotphase
        @series begin
            inds = findall(x -> x == 0, frd.w)
            subplot --> 2
            useinds = setdiff(1:length(frd.w), inds)
            (hz ? 1 / (2π) : 1) .* frd.w[useinds], unwrap(angle.(frd.r[useinds]))
        end
    end
    nothing
end

freqvec(h, k) = LinRange(0, π / h, length(k))

"""
    H, N = tfest(data, σ = 0.05)

Estimate a transfer function model using the Correlogram approach.
    Both `H` and `N` are of type `FRD` (frequency-response data).

`σ` determines the width of the Gaussian window applied to the estimated correlation functions before FFT. A larger `σ` implies less smoothing.
- `H` = Syu/Suu             Process transfer function
- `N` = Sy - |Syu|²/Suu     Noise PSD
"""
function tfest(d, σ::Real = 0.05)
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



@userplot Coherenceplot
"""
    coherenceplot(d, [(;n=..., noverlap=...); hz=false)

Calculates and plots the (squared) coherence Function κ. κ close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy.

`hz` indicates Hertz instead of rad/s

Keyword arguments to `coherence` are supplied as a named tuple as a second positional argument .
"""
coherenceplot

@recipe function coherenceplot(p::Coherenceplot; hz = false)
    ae =
        ArgumentError("Call like this: coherenceplot(iddata; hz=false)")
    d = p.args[1]
    d isa AbstractIdData || throw(ae)
    if length(p.args) >= 2
        kwargs = p.args[2]
    else
        kwargs = NamedTuple()
    end
    y, u, h = output(d), input(d), sampletime(d)
    yscale --> :identity
    xscale --> :log10
    ylims --> (0, 1)
    xguide --> (hz ? "Frequency [Hz]" : "Frequency [rad/s]")
    title --> "Coherence"
    legend --> false
    frd = coherence(d; kwargs...)
    @series begin
        inds = findall(x -> x == 0, frd.w)
        useinds = setdiff(1:length(frd.w), inds)
        (hz ? 1 / (2π) : 1) .* frd.w[useinds], abs.(frd.r[useinds])
    end
    nothing
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

@userplot Impulseestplot

"""
    impulseestplot(data,n)

Estimates the system impulse response by fitting an `n`:th order FIR model and plots the result with a 95% confidence band.
See also `impulseestplot`
"""
impulseestplot
@recipe function impulseestplot(p::Impulseestplot; λ = 0)
    d = p.args[1]
    n = length(p.args) >= 2 ? p.args[2] : 25
    ir, t, Σ = impulseest(d, n; λ = λ)
    title --> "Estimated Impulse Response"
    xguide --> "Time [s]"

    @series begin
        label --> ""
        t, ir
    end
    linestyle := :dash
    color := :black
    label := ""
    seriestype := :hline
    @series begin
        t, 2 .* sqrt.(diag(Σ))
    end
    @series begin
        t, -2 .* sqrt.(diag(Σ))
    end
end


@userplot Crosscorplot
@recipe function crosscorplot(p::Crosscorplot)
    d = p.args[1]
    N = length(d)
    lags = length(p.args) >= 2 ? p.args[2] : -max(N ÷ 10, 100):max(N ÷ 2, 100)
    xc = crosscor(time1(d.u), time1(d.y), lags, demean = true)
    title --> "Input-Output cross correlation"
    xguide --> "Lag [s]"

    @series begin
        seriestype --> :sticks
        label --> ""
        lags .* d.Ts, xc
    end
    linestyle := :dash

    seriescolor := :black
    label := ""
    primary := false
    # Ni = N .- abs.(lags)
    @series begin
        seriestype := :hline
        # lags.*d.Ts, 2 .*sqrt.(1 ./ Ni) # The denominator in crosscorr already takes care of this
        [2 .* sqrt.(1 ./ N)]
    end
    @series begin
        seriestype := :hline
        # lags.*d.Ts, -2 .*sqrt.(1 ./ Ni)
        [-2 .* sqrt.(1 ./ N)]
    end
end


function ControlSystems.gangoffour(P::FRD, C::FRD, ω = nothing)
    ω === nothing || ω == P.ω || error("Incosistent frequency vectors")
    S = sensitivity(P, C)
    D = (P * S)
    N = (C * S)
    T = (P * N)
    return S, D, N, T
end
