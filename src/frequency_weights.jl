function get_frequencyweight_tf(responsetype::FilterType)
    designmethod = Butterworth(2)
    digitalfilter(responsetype, designmethod)
end

get_frequencyweight_tf(G::FilterCoefficients) = G

function frequency_weight(system, N)
    ω = range(0, stop=pi, length=N)
    G = get_frequencyweight_tf(system)
    Φ = freqz(G, ω)
    acf = irfft(abs2.(Φ), 2length(Φ)-1)[1:end÷2+1]
    n = length(acf)
    W = similar(acf, n, n)
    for di = -n+1:n-1
        W[diagind(W,di)] .= acf[abs(di)+1]
    end
    Symmetric(W)
end

weighted_estimator(H::FilterType) = (A,y) -> wls(A,y,frequency_weight(H, size(y,1)))
