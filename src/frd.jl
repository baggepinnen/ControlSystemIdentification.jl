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
import Base: +, -, *, length, sqrt
Base.vec(f::FRD) = f.r
*(f::FRD, f2)    = FRD(f.w, f.r .* vec(f2))
+(f::FRD, f2)    = FRD(f.w, f.r .+ vec(f2))
-(f::FRD, f2)    = FRD(f.w, f.r .- vec(f2))
-(f::FRD)        = FRD(f.w, -f.r)
length(f::FRD)   = length(f.w)
sqrt(f::FRD)     = FRD(f.w, sqrt.(f.r))

feedback(P::FRD,K) = FRD(P.w,vec(P)./(1. .+ vec(P).*vec(K)))
feedback(P,K::FRD) = FRD(K.w,vec(P)./(1. .+ vec(P).*vec(K)))
feedback(P::FRD,K::FRD) = feedback(P,vec(K))

feedback(P::FRD,K::LTISystem) = feedback(P,freqresp(K,P.w)[:,1,1])
feedback(P::LTISystem,K::FRD) = feedback(freqresp(P,K.w)[:,1,1],K)

@recipe function plot_frd(frd::FRD)
    yscale --> :log10
    xscale --> :log10
    xlabel --> "Frequency [rad/s]"
    ylabel --> "Magnitude"
    title --> "Bode Plot"
    legend --> false
    @series begin
        inds = findall(x->x==0, frd.w)
        useinds = setdiff(1:length(frd.w), inds)
        frd.w[useinds], abs.(frd.r[useinds])
    end
    nothing
end

freqvec(h,k) = LinRange(0,π/h, length(k))

"""
    H, N = tfest(h,y,u, σ = 0.05)

Estimate a transfer function model using the Correlogram approach.
    Both `H` and `N` are of type `FRD` (frequency-response data).

`σ` determines the width of the Gaussian window applied to the estimated correlation functions before FFT.
- `H` = Syu/Suu             Process transfer function
- `N` = Sy - |Syu|²/Suu     Noise PSD
"""
function tfest(h::Real,y::AbstractVector,u::AbstractVector, σ = 0.05)
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

Calculates the coherence Function. κ close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy. An estimated noise model is also returned.
κ: Coherence function (not squared)
N: Noise model
"""
function coherence(h::Real,y::AbstractVector,u::AbstractVector; n = length(y)÷10, noverlap = n÷2, window=hamming)
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
coherenceplot(h,y,u)

Calculates and plots the coherence Function κ. κ close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy.
"""
coherenceplot

@recipe function cp(p::Coherenceplot)
    ae = ArgumentError("Call like this: coherenceplot(h,y,u), where h is sample time and y/u are vectors of equal length.")
    length(p.args) == 3 || throw(ae)
    h,y,u = p.args[1:3]
    length(u) == length(y) || throw(ae)
    h isa Number || throw(ae)
    yscale --> :identity
    xscale --> :log10
    ylims --> (0,1)
    xlabel --> "Frequency [rad/s]"
    title --> "Coherence"
    legend --> false
    frd = coherence(h,y,u)
    @series begin
        inds = findall(x->x==0, frd.w)
        useinds = setdiff(1:length(frd.w), inds)
        frd.w[useinds], abs.(frd.r[useinds])
    end
    nothing
end

"""
ir, t, Σ = impulseest(h,y,u,n)

Estimates the system impulse response by fitting an `n`:th order FIR model. Returns impulse-response estimate, time vector and covariance matrix.
See also `impulseestplot`
"""
function impulseest(h,y,u,n,λ=0)
    yt,A = getARXregressor(y,u,0,n)
    ir = ls(A,yt,λ)
    t = range(h,length=n, step=h)
    Σ = parameter_covariance(yt, A, ir, λ)
    ir./h, t, Σ
end

@userplot Impulseestplot

"""
impulseestplot(h,y,u,n)

Estimates the system impulse response by fitting an `n`:th order FIR model and plots the result with a 95% confidence band.
See also `impulseestplot`
"""
impulseestplot
@recipe function impulseestplot(p::Impulseestplot)
    h,y,u = p.args[1:3]
    n = length(p.args) >= 4 ? p.args[4] : 25
    λ = length(p.args) >= 5 ? p.args[5] : 0
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
