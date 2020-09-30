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


    lastlarge = findlast(x->abs(x) > sqrt(eps())*acf[1], acf)
    if lastlarge === nothing || lastlarge == n
        lastlarge = n-1
    end
    bands = [0=>Fill(acf[1], n)]
    for i = 1:lastlarge
        slice = Fill(acf[i+1], n-i)
        push!(bands,  i=>slice)
        push!(bands, -i=>slice)
    end
    W = BandedMatrix((bands...,), (n,n))

    # W = similar(acf, n, n)
    # for di = -n+1:n-1
    #     W[diagind(W,di)] .= acf[abs(di)+1]
    # end
    # Symmetric(W)
end

weighted_estimator(H::FilterType) = (A,y) -> wls(A,y,frequency_weight(H, size(y,1)))
