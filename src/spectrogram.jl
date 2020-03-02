import DSP.Periodograms: PSDOnly,nextfastfft,Spectrogram,spectrogram,stft,fft2pow!,fft2oneortwosided!, compute_window,arraysplit,stfttype

function DSP.spectrogram(s::AbstractVector{T}, estimator::Function, n::Int=length(s)>>3, noverlap::Int=n>>1;
                     onesided::Bool=eltype(s)<:Real,
                     nfft::Int=nextfastfft(n), fs::Real=1,
                     window::Union{Function,AbstractVector,Nothing}=nothing) where T

    out = stft(s, estimator, n, noverlap, PSDOnly(); onesided=onesided, nfft=nfft, fs=fs, window=window)
    Spectrogram(out, onesided ? DSP.rfftfreq(nfft, fs) : DSP.fftfreq(nfft, fs),
                (n/2 : n-noverlap : (size(out,2)-1)*(n-noverlap)+n/2) / fs)

end

function stft(s::AbstractVector{T}, estimator::Function, n::Int=length(s)>>3, noverlap::Int=n>>1,
              psdonly::Union{Nothing,PSDOnly}=nothing;
              onesided::Bool=eltype(s)<:Real, nfft::Int=nextfastfft(n), fs::Real=1,
              window::Union{Function,AbstractVector,Nothing}=nothing) where T


    win, norm2 = compute_window(window, n)
    sig_split = arraysplit(s, n, noverlap, nfft, win)
    nout = onesided ? (nfft >> 1)+1 : nfft
    out = zeros(stfttype(T, psdonly), nout, length(sig_split))

    freqs = onesided ? DSP.rfftfreq(nfft, fs) : DSP.fftfreq(nfft, fs)
    r = fs*norm2/sqrt(length(freqs))
    offset = 0
    for (i,sig) in enumerate(sig_split)
        # mul!(tmp, plan, sig)
        tmp = estimator(sig, freqs)
        if isa(psdonly, PSDOnly)
            fft2pow!(out, tmp, nfft, r, onesided, offset)
        else
            fft2oneortwosided!(out, tmp, nfft, onesided, offset)
        end
        offset += nout
    end
    out
end

"""
    model_spectrum(f, h, args...; kwargs...)

DOCSTRING

#Arguments:
- `f`: the model-estimation function, e.g., `ar,arma`
- `h`: The sample time
- `args`: arguments to `f`
- `kwargs`: keyword arguments to `f`

# Example:
```
using ControlSystemIdentification, DSP
T = 1000
s = sin.((1:T) .* 2pi/10)
S1 = spectrogram(s,window=hanning)
estimator = model_spectrum(ar,1,2)
S2 = spectrogram(s,estimator,window=rect)
plot(plot(S1),plot(S2)) # Requires the package LPVSpectral.jl
```
"""
function model_spectrum(f,h,args...;kwargs...)
    function (s::AbstractArray{T}, freqs) where T
        d = iddata(s,h)
        model = f(d, args...;kwargs...)
        tmp = vec(Complex{T}.(freqresp(model,T(2pi) .* freqs)))
    end
end






# function rootspectrogram(model, fs)
#     roots = map(1:length(model)) do i
#         sys = tf(1,[1;-reverse(model.θ[:,i])],1)
#         (sort(pole(sys), by=imag, rev=true)[1:end÷2])
#         # log.(sort(complex.(eigvals(model.At[:,:,i])), by=imag, rev=true)[1:end÷2])
#     end
#     S = reduce(hcat,roots)
#     fs/(2pi) .* angle.(S)'
# end
