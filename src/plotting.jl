@recipe function plot(d::AbstractIdData)
    y = time1(output(d))
    n = noutputs(d)
    if hasinput(d)
        u = time1(input(d))
        n += ninputs(d)
    end
    layout --> (n, 1)
    legend --> false
    xguide --> "Time"
    link --> :x
    xvec = range(0, step = sampletime(d), length = length(d))

    for i = 1:size(y, 2)
        @series begin
            title --> "Output $i"
            label --> "Output $i"
            xvec, y[:, i]
        end
    end
    if hasinput(d)
        for i = 1:size(u, 2)
            @series begin
                title --> "Input $i"
                label --> "Input $i"
                xvec, u[:, i]
            end
        end
    end
end

@recipe function plot_spectrogram(::Type{Val{:specplot}}, p)
    seriestype := :heatmap
    title --> "Spectrogram"
    yscale --> :log10
    yguide --> "Frequency [Hz]"
    xguide --> "Time [s]"
    p.time, p.freq[2:end], log.(p.power)[2:end,:]
end

@userplot Specplot

@recipe function specplot(p::Specplot)
    d = p.args[1]
    d isa ControlSystemIdentification.AbstractIdData || throw(ArgumentError("Expected AbstractIdData"))
    ny = d.ny
    nu = d.nu
    layout --> (ny+nu, 1)
    link --> :x
    for i = 1:ny
        S = spectrogram(d.y[i,:], p.args[2:end]...; fs=d.fs, window=hanning)
        @series begin
            seriesstyle := :specplot
            subplot := i
            title --> "Output $i"
            S
        end
    end
    for i = 1:nu
        S = spectrogram(d.u[i,:], p.args[2:end]...; fs=d.fs, window=hanning)
        @series begin
            seriesstyle := :specplot
            subplot := ny + i
            title --> "Input $i"
            S
        end
    end
end



@userplot Simplot
"""
	simplot(sys, data, x0=nothing; ploty=true, plote=false)

Plot system simulation and measured output to compare them.
`ploty` determines whether or not to plot the measured signal
`plote` determines whether or not to plot the residual
"""
simplot
@recipe function simplot(p::Simplot; ploty = true, plote = false)
    sys, d = p.args[1:2]
    y = oftype(randn(2, 2), output(d))
    x0 = length(p.args) > 2 ? p.args[3] : nothing
    x0 = get_x0(x0, sys, d)
    yh = simulate(sys, d, x0)
    xguide --> "Time [s]"
    yguide --> "Output"
    t = timevec(d)
    err = nrmse(y, yh)
    ploty && @series begin
        label --> ["y$i" for i in (1:d.ny)']
        t, y'
    end
    @series begin
        label --> ["sim fit $i :$(round(err, digits=2))%" for (i,err) in enumerate(err')]
        t, yh'
    end
    plote && @series begin
        label --> "sim resid."
        t, y' - yh'
    end
    nothing
end


@userplot Predplot
"""
	predplot(sys, data, x0=nothing; ploty=true, plote=false)

Plot system simulation and measured output to compare them.
`ploty` determines whether or not to plot the measured signal
`plote` determines whether or not to plot the residual
"""
predplot
@recipe function predplot(p::Predplot; ploty = true, plote = false)
    sys, d = p.args[1:2]
    y = oftype(randn(2, 2), output(d))
    u = oftype(randn(2, 2), input(d))
    x0 = length(p.args) > 2 ? p.args[3] : :estimate
    x0 = get_x0(x0, sys, d)
    yh = predict(sys, y, u, x0)
    xguide --> "Time [s]"
    yguide --> "Output"
    t = timevec(d)
    err = nrmse(y, yh)
    ploty && @series begin
        label --> ["y$i" for i in (1:d.ny)']
        t, y'
    end
    @series begin
        label --> ["pred fit $i :$(round(err, digits=2))%" for (i,err) in enumerate(err')]
        t, yh'
    end
    plote && @series begin
        label --> "pred resid."
        t, y' - yh'
    end
    nothing
end


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
    ninputs(d) == 1 || throw(ArgumentError("coherenceplot only supports a single input. Index the data object like `d[i,j]` to obtain the `i`:th output and the `j`:th input."))
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
    for i = 1:d.ny
        frd = coherence(d[i,1]; kwargs...)
        @series begin
            inds = findall(x -> x == 0, frd.w)
            useinds = setdiff(1:length(frd.w), inds)
            label --> (d.ny == 1 ? "" : "To y_$i")
            (hz ? 1 / (2π) : 1) .* frd.w[useinds], abs.(frd.r[useinds])
        end
    end
    nothing
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

"""
    crosscorplot(data, [lags])

Plot the cross correlation betweein input and output for `lags` that default to 10% of the length of the dataset on the negative side and 50% on the positive side but no more than 100 on each side.
"""
crosscorplot

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


@userplot Find_na
"""
    find_na(y::AbstractVector,n::Int)
Plots the RMSE and AIC For model orders up to `n`. Useful for model selection
"""
find_na
@recipe function find_na(p::Find_na)
    y, n = p.args[1:2]
    y = time1(y)
    error = zeros(n, 2)
    for i = 1:n
        yt, A = getARXregressor(y, 0y, i, 0)
        e = yt - A * (A \ yt)
        error[i, 1] = rms(e)
        error[i, 2] = aic(e, i)
    end
    layout --> 2
    title --> ["RMS error" "AIC"]
    seriestype --> :scatter
    @series begin
        error
    end
end

@userplot Find_nanb
"""
    find_nanb(d::InputOutputData,na,nb)
Plots the RMSE and AIC For model orders up to `n`. Useful for model selection
"""
find_nanb
@recipe function find_nanb(p::Find_nanb; logrms = false)
    d, na, nb = p.args[1:3]
    y, u = time1(output(d)), time1(input(d))
    error = zeros(na, nb, 2)
    for i = 1:na, j = 1:nb
        yt, A = getARXregressor(y, u, i, j)
        e = yt - A * (A \ yt)
        error[i, j, 1] = logrms ? log10.(rms(e)) : rms(e)
        error[i, j, 2] = aic(e, i + j)
    end
    layout --> 2
    seriestype --> :heatmap
    xticks := (1:nb, 1:nb)
    yticks := (1:na, 1:na)
    yguide := "na"
    xguide := "nb"
    @series begin
        title := "RMS error"
        subplot := 1
        error[:, :, 1]
    end
    @series begin
        title := "AIC"
        subplot := 2
        error[:, :, 2]
    end
end