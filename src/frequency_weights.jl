function get_frequencyweight_tf(responsetype::FilterType; fs)
    designmethod = Butterworth(2)
    digitalfilter(responsetype, designmethod; fs=fs)
end

get_frequencyweight_tf(G::FilterCoefficients) = G

function frequency_weight(system, N; fs=2)
    ω = range(0, stop = pi, length = N)
    G = get_frequencyweight_tf(system; fs)
    Φ = DSP.freqresp(G, ω)
    acf = irfft(abs2.(Φ), 2length(Φ) - 1)[1:end÷2+1]
    n = length(acf)


    # lastlarge = findlast(x -> abs(x) > sqrt(eps()) * acf[1], acf)
    # if lastlarge === nothing || lastlarge == n
    #     lastlarge = n - 1
    # end
    # bands = [0 => Fill(acf[1], n)]
    # for i = 1:lastlarge
    #     slice = Fill(acf[i+1], n - i)
    #     push!(bands, i => slice)
    #     push!(bands, -i => slice)
    # end
    # W = BandedMatrix((bands...,), (n, n))

    W = similar(acf, n, n)
    for di = -n+1:n-1
        W[diagind(W,di)] .= acf[abs(di)+1]
    end
    Symmetric(W)
end

weighted_estimator(H::FilterType; fs=2) = (A, y) -> wls(A, y, frequency_weight(H, size(y, 1); fs))

"""
    prefilter(d::AbstractIdData, responsetype::FilterType)

Filter both input and output of the identification data using zero-phase filtering (`filtfilt`).
Since both input and output is filtered, linear identification will not be affected in any other way than to focus the fit on the selected frequency range, i.e. the range that has high gain in the provided filter. Note, if the system that generated `d` is nonlinear, identification might be severely impacted by this transformation. Verify linearity with, e.g., [`coherenceplot`](@ref).
"""
function prefilter(d::AbstractIdData, responsetype::FilterType)
    fs = 1/d.Ts
    H = get_frequencyweight_tf(responsetype; fs=fs)
    y = time1(output(d))
    u = time1(input(d))
    y = filtfilt(H, y)
    u = filtfilt(H, u)
    iddata(y', u', d.Ts)
end

"""
    prefilter(d::AbstractIdData, l::Number, u::Number)

Filter input and output with a bandpass filter between `l` and `u` Hz. If `l = 0` a lowpass filter will be used, and if `u = Inf` a highpass filter will be used.
"""
function prefilter(d::AbstractIdData, l::Number, u::Number)
    responsetype = if u == Inf
        Highpass(l)
    elseif l <= 0
        Lowpass(u)
    else
        Bandpass(l, u)
    end
    prefilter(d, responsetype)
end

"""
    prefilter(f, d::InputOutputData)

Apply filter coefficients to identification data
"""
function prefilter(f, d::InputOutputData)
    u = filt(f, d.u')'
    y = filt(f, d.y')'
    iddata(y, u, d.Ts)
end