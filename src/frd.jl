import ControlSystems: feedback

struct FRD{WT<:AbstractVector,RT<:AbstractVector} <: LTISystem
    w::WT
    r::RT
end

FRD(w, s::LTISystem) = FRD(w, freqresp(s,w)[:,1,1])
import Base: +, -, *, length
Base.vec(f::FRD) = f.r
*(f::FRD, f2)    = FRD(f.w, f.r .* vec(f2))
+(f::FRD, f2)    = FRD(f.w, f.r .+ vec(f2))
-(f::FRD, f2)    = FRD(f.w, f.r .- vec(f2))
-(f::FRD)        = FRD(f.w, -f.r)
length(f::FRD) = length(f.w)

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



"""
    H, N = tfest(h,y,u)

Estimate a transfer function model using the Correlogram approach
H = Syu/Suu             Process transfer function
N = Sy - |Syu|²/Suu     Noise model
"""
function tfest(h,y,u)
    error("Not implemented yet.")
    Cyu = xcorr(y,u)
    Cyy = xcorr(y,y)
    Cuu = xcorr(u,u)
    Syu = welch_pgram(Cyu, fs=2π/h, window=DSP.Windows.hamming) # TODO: this must result in complex vector, i.e., do not use pgram
    Syy = welch_pgram(Cyy, fs=2π/h, window=DSP.Windows.hamming)
    Suu = welch_pgram(Cuu, fs=2π/h, window=DSP.Windows.hamming)
    H = FRD(Syu.freq, Syu.power./Suu.power)
    N = FRD(Syu.freq, Syy.power .- abs2.(Syu.power)./Suu.power)
    return H, N
end



"""
    κ, N = coherence(h,y,u)

Calculates the coherence Function. κ close to 1 indicates a good explainability of energy in the output signal by energy in the input signal. κ << 1 indicates that either the system is nonlinear, or a strong noise contributes to the output energy. An estimated noise model is also returned.
κ: Coherence function (not squared)
N: Noise model
"""
function coherence(h,y,u; n = length(y)÷10, noverlap = n÷2, window=hamming)
    Syy     = zeros(length(y)÷10)
    Suu     = zeros(length(y)÷10)
    Syu     = zeros(ComplexF64,length(y)÷10)
    win, norm2 = DSP.Periodograms.compute_window(window, n)
    uw = arraysplit(u,n,noverlap,nextfastfft(n),win)
    for (i,y) in enumerate(arraysplit(y,n,noverlap,nextfastfft(n),win))
        u       = uw[i]
        xy      = fft(y)
        xu      = fft(u)
        # Cross spectrum
        Syu .+= xy.*conj.(xu)
        Syy .+= abs2.(xy)
        Suu .+= abs2.(xu)
    end
    k = (abs2.(Syu)./(Suu.*Syy))[end÷2+1:end]
    Sch = FRD(LinRange(0,π/h, length(k)),k)
    return Sch
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
    ylabel --> "Coherence"
    title --> ""
    legend --> false
    frd = coherence(h,y,u)
    @series begin
        inds = findall(x->x==0, frd.w)
        useinds = setdiff(1:length(frd.w), inds)
        frd.w[useinds], abs.(frd.r[useinds])
    end
    nothing
end
