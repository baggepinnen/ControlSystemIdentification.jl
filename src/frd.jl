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

FRD(w, s::LTISystem) = FRD(w, freqresp(s,w)[:,1,1])
import Base: +, -, *, length, sqrt, getindex
Base.vec(f::FRD) = f.r
*(f::FRD, f2)    = FRD(f.w, f.r .* vec(f2))
+(f::FRD, f2)    = FRD(f.w, f.r .+ vec(f2))
-(f::FRD, f2)    = FRD(f.w, f.r .- vec(f2))
-(f::FRD)        = FRD(f.w, -f.r)
length(f::FRD)   = length(f.w)
Base.size(f::FRD)   = (1,1) # Size in the ControlSystems sense
Base.lastindex(f::FRD) = length(f)
function Base.getproperty(f::FRD, s::Symbol)
    s === :Ts && return 1/((f.w[2]-f.w[2])/(2π))
    getfield(f,s)
end
ControlSystems.noutputs(f::FRD) = 1
ControlSystems.ninputs(f::FRD) = 1

sqrt(f::FRD)     = FRD(f.w, sqrt.(f.r))
getindex(f::FRD, i) = FRD(f.w[i], f.r[i])
getindex(f::FRD, i::Int) = f.r[i]
getindex(f::FRD, i::Int, j::Int) = (@assert(i==1 && j==1); f)

feedback(P::FRD,K) = FRD(P.w,vec(P)./(1. .+ vec(P).*vec(K)))
feedback(P,K::FRD) = FRD(K.w,vec(P)./(1. .+ vec(P).*vec(K)))
feedback(P::FRD,K::FRD) = feedback(P,vec(K))

feedback(P::FRD,K::LTISystem) = feedback(P,freqresp(K,P.w)[:,1,1])
feedback(P::LTISystem,K::FRD) = feedback(freqresp(P,K.w)[:,1,1],K)

@recipe function plot_frd(frd::FRD; hz=false)
    yscale --> :log10
    xscale --> :log10
    xlabel --> (hz ? "Frequency [Hz]" : "Frequency [rad/s]")
    ylabel --> "Magnitude"
    title --> "Bode Plot"
    legend --> false
    @series begin
        inds = findall(x->x==0, frd.w)
        useinds = setdiff(1:length(frd.w), inds)
        (hz ? 1/(2π) : 1) .* frd.w[useinds], abs.(frd.r[useinds])
    end
    nothing
end

freqvec(h,k) = LinRange(0,π/h, length(k))

"""
    H, N = tfest(data, σ = 0.05)

Estimate a transfer function model using the Correlogram approach.
    Both `H` and `N` are of type `FRD` (frequency-response data).

`σ` determines the width of the Gaussian window applied to the estimated correlation functions before FFT.
- `H` = Syu/Suu             Process transfer function
- `N` = Sy - |Syu|²/Suu     Noise PSD
"""
function tfest(d, σ = 0.05)
    y,u,h = time1(output(d)),time1(input(d)),sampletime(d)
    Syy,Suu,Syu = fft_corr(y,u,σ)
    w   = freqvec(h,Syu)
    H   = FRD(w, Syu./Suu)
    N   = FRD(w, @.(Syy - abs2(Syu)/Suu)./length(y))
    return H, N
end

function fft_corr(y,u,σ = 0.05)
    n = length(y)
    w = gaussian(2n-1,σ)

    Syu = rfft(ifftshift(w.*xcorr(y,u)))
    Syy = rfft(ifftshift(w.*xcorr(y,y)))
    Suu = rfft(ifftshift(w.*xcorr(u,u)))
    Syy,Suu,Syu
end



"""
    κ, N = coherence(h,y,u)
    κ, N = coherence(d)

Calculates the coherence Function. κ close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy. An estimated noise model is also returned.
κ: Coherence function (not squared)
N: Noise model
"""
function coherence(d; n = length(d)÷10, noverlap = n÷2, window=hamming)
    y,u,h = time1(output(d)), time1(input(d)), sampletime(d)
    Syy,Suu,Syu = wcfft(y,u,n=n,noverlap=noverlap,window=window)
    k = (abs2.(Syu)./(Suu.*Syy))#[end÷2+1:end]
    Sch = FRD(freqvec(h,k),k)
    return Sch
end

function wcfft(y,u; n = length(y)÷10, noverlap = n÷2, window=hamming)
    win, norm2 = DSP.Periodograms.compute_window(window, n)
    uw  = arraysplit(u,n,noverlap,nextfastfft(n),win)
    yw  = arraysplit(y,n,noverlap,nextfastfft(n),win)
    Syy = zeros(length(uw[1])÷2 + 1)
    Suu = zeros(length(uw[1])÷2 + 1)
    Syu = zeros(ComplexF64,length(uw[1])÷2 + 1)
    for i in eachindex(uw)
        xy      = rfft(yw[i])
        xu      = rfft(uw[i])
        Syu .+= xy.*conj.(xu)
        Syy .+= abs2.(xy)
        Suu .+= abs2.(xu)
    end
    Syy,Suu,Syu
end

function wfft_corr(y,u; n = length(y)÷10, noverlap = n÷2, window=hamming)
    win, norm2 = DSP.Periodograms.compute_window(window, n)
    Cyu = xcorr(y,u)
    Cyy = xcorr(y,y)
    Cuu = xcorr(u,u)
    uw  = arraysplit(Cuu,n,noverlap,nextfastfft(n),win)
    yw  = arraysplit(Cyy,n,noverlap,nextfastfft(n),win)
    yuw = arraysplit(Cyu,n,noverlap,nextfastfft(n),win)
    Syy = zeros(ComplexF64,length(uw[1])÷2 + 1)
    Suu = zeros(ComplexF64,length(uw[1])÷2 + 1)
    Syu = zeros(ComplexF64,length(uw[1])÷2 + 1)
    for i in eachindex(uw)
        xy    = rfft(yw[i])
        xu    = rfft(uw[i])
        xyu   = rfft(yuw[i])
        Syu .+= xyu
        Syy .+= xy
        Suu .+= xu
    end
    Syy,Suu,Syu
end


@userplot Coherenceplot
"""
coherenceplot(d; hz=false)

Calculates and plots the coherence Function κ. κ close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy.

`hz` indicates Hertz instead of rad/s
"""
coherenceplot

@recipe function cp(p::Coherenceplot; hz=false)
    ae = ArgumentError("Call like this: coherenceplot(iddata), where h is sample time and y/u are vectors of equal length.")
    d = p.args[1]
    d isa AbstractIdData || throw(ae)
    if length(p.args) >= 2
        kwargs = p.args[2]
    else
        kwargs = NamedTuple()
    end
    y,u,h = output(d), input(d), sampletime(d)
    yscale --> :identity
    xscale --> :log10
    ylims --> (0,1)
    xlabel --> (hz ? "Frequency [Hz]" : "Frequency [rad/s]")
    title --> "Coherence"
    legend --> false
    frd = coherence(d; kwargs...)
    @series begin
        inds = findall(x->x==0, frd.w)
        useinds = setdiff(1:length(frd.w), inds)
        (hz ? 1/(2π) : 1) .* frd.w[useinds], abs.(frd.r[useinds])
    end
    nothing
end

"""
ir, t, Σ = impulseest(h,y,u,n)

Estimates the system impulse response by fitting an `n`:th order FIR model. Returns impulse-response estimate, time vector and covariance matrix.
See also `impulseestplot`
"""
function impulseest(h,y,u,n,λ=0)
    N = min(length(u),length(y))
    @views yt,A = getARXregressor(y[1:N],u[1:N],0,n)
    ir = ls(A,yt,λ)
    t = range(h,length=n, step=h)
    Σ = parameter_covariance(yt, A, ir, λ)
    ir./h, t, Σ
end

@userplot Impulseestplot

"""
impulseestplot(data,n)

Estimates the system impulse response by fitting an `n`:th order FIR model and plots the result with a 95% confidence band.
See also `impulseestplot`
"""
impulseestplot
@recipe function impulseestplot(p::Impulseestplot)
    d = p.args[1]
    y,u,h = output(d), input(d), sampletime(d)
    n = length(p.args) >= 2 ? p.args[2] : 25
    λ = length(p.args) >= 3 ? p.args[3] : 0
    ir,t,Σ = impulseest(h,y,u,n,λ)
    title --> "Estimated Impulse Response"
    xlabel --> "Time [s]"

    @series begin
        label --> ""
        t,ir
    end
    linestyle := :dash
    color := :black
    label := ""
    seriestype := :hline
    @series begin
        t, 2 .*sqrt.(diag(Σ))
    end
    @series begin
        t, -2 .*sqrt.(diag(Σ))
    end
end
