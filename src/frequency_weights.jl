function get_frequencyweight_tf(responsetype::FilterType)
    designmethod = Butterworth(2)
    digitalfilter(responsetype, designmethod)
end

get_frequencyweight_tf(G::FilterCoefficients) = G

function frequency_weight(system, N)
    ω = range(0, stop = pi, length = N)
    G = get_frequencyweight_tf(system)
    Φ = freqz(G, ω)
    acf = irfft(abs2.(Φ), 2length(Φ) - 1)[1:end÷2+1]
    n = length(acf)


    lastlarge = findlast(x -> abs(x) > sqrt(eps()) * acf[1], acf)
    if lastlarge === nothing || lastlarge == n
        lastlarge = n - 1
    end
    bands = [0 => Fill(acf[1], n)]
    for i = 1:lastlarge
        slice = Fill(acf[i+1], n - i)
        push!(bands, i => slice)
        push!(bands, -i => slice)
    end
    W = BandedMatrix((bands...,), (n, n))

    # W = similar(acf, n, n)
    # for di = -n+1:n-1
    #     W[diagind(W,di)] .= acf[abs(di)+1]
    # end
    # Symmetric(W)
end

weighted_estimator(H::FilterType) = (A, y) -> wls(A, y, frequency_weight(H, size(y, 1)))

"""
    frequency_focus(d::AbstractIdData, responsetype::FilterType)

Filter both input and output of the identification data using zero-phase filtering (`filtfilt`).
Since both input and output is filtered, linear identification will not be affected in any other way than to focus the fit on the selected frequency range, i.e. the range that has high gain in the provided filter. Note, if the system that generated `d` is nonlinear, identification might be severely impacted by this transformation. Verify linearity with, e.g., `coherenceplot`.
"""
function frequency_focus(d::AbstractIdData, responsetype::FilterType)
    H = get_frequencyweight_tf(responsetype)
    y = time1(output(d))
    u = time1(input(d))
    y = filtfilt(H, y)
    u = filtfilt(H, u)
    iddata(y', u', d.Ts)
end

"""
    frequency_focus(d::AbstractIdData, l::Number, u::Number)

Filter input and output with a bandpass filter between `l` and `u` Hz. If `l = 0` a lowpass filter will be used, and if `u = Inf` a highpass filter will be used.
"""
function frequency_focus(d::AbstractIdData, l::Number, u::Number)
    responsetype = if u == Inf
        Highpass(l, fs = d.fs)
    elseif l <= 0
        Lowpass(u, fs = d.fs)
    else
        Bandpass(l, u, fs = d.fs)
    end
    frequency_focus(d, responsetype)
end