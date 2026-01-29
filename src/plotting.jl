@recipe function plot(d::AbstractIdData; ploty=true, plotu=true)
    y = time1(output(d))
    n = ploty ? noutputs(d) : 0
    if plotu && hasinput(d)
        u = time1(input(d))
        n += ninputs(d)
    end
    layout --> (n, 1)
    label --> ""
    xguide --> "Time"
    link --> :x
    xvec = range(0, step = sampletime(d), length = length(d))

    if ploty
        for i = 1:size(y, 2)
            @series begin
                title --> "Output $i"
                label --> "Output $i"
                xvec, y[:, i]
            end
        end
    end
    if plotu && hasinput(d)
        for i = 1:size(u, 2)
            @series begin
                title --> "Input $i"
                label --> "Input $i"
                xvec, u[:, i]
            end
        end
    end
end

@recipe function plot_spectrogram(p::DSP.Periodograms.Spectrogram)
    seriestype := :heatmap
    title --> "Spectrogram"
    yscale --> :log10
    yguide --> "Frequency [Hz]"
    xguide --> "Time [s]"
    p.time, p.freq[2:end], log.(p.power)[2:end,:]
end

@userplot Specplot

"""
    specplot(d::IdData, args...; kwargs...)

Plot a spectrogram of the input and output timeseries. See also [`welchplot`](@ref).

Additional arguments are passed along to `DSP.spectrogram`.
"""
specplot
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


@userplot Welchplot

"""
    welchplot(d::IdData, args...; kwargs...)

Plot a Wlch peridogram of the input and output timeseries. See also [`specplot`](@ref).

Additional arguments are passed along to `DSP.welch_pgram`.
"""
welchplot
@recipe function welchplot(p::Welchplot)
    d = p.args[1]
    d isa ControlSystemIdentification.AbstractIdData || throw(ArgumentError("Expected AbstractIdData"))
    ny = d.ny
    nu = d.nu
    link --> :both
    for i = 1:ny
        S = DSP.welch_pgram(d.y[i,:], p.args[2:end]...; fs=d.fs, window=hanning)
        @series begin
            xscale --> :log10
            yscale --> :log10
            lab --> "Output $i"
            xguide --> "Frequency [Hz]"
            S.freq[2:end], S.power[2:end]
        end
    end
    for i = 1:nu
        S = DSP.welch_pgram(d.u[i,:], p.args[2:end]...; fs=d.fs, window=hanning)
        @series begin
            xscale --> :log10
            yscale --> :log10
            lab --> "Input $i"
            xguide --> "Frequency [Hz]"
            S.freq[2:end], S.power[2:end]
        end
    end
end


_process_simplotargs(sys, d::AbstractIdData, x0 = :estimate) = sys, d, get_x0(x0, sys, d)

_process_simplotargs(d::AbstractIdData, sys, x0 = :estimate) = _process_simplotargs(sys, d, x0) # sort arguments

# function _process_simplotargs(d::AbstractIdData, systems::LTISystem...) 
#     map(systems) do sys
#         _process_simplotargs(sys, d, :estimate)
#     end
# end

@userplot Simplot
"""
	simplot(sys, data, x0=:estimate; ploty=true, plote=false, sysname="")

Plot system simulation and measured output to compare them.

By default, the initial condition `x0` is estimated using the data. To start the simulation from the origin, provide `x0 = :zero` or `x0 = zeros(sys.nx)`.

- `ploty` determines whether or not to plot the measured signal
- `plote` determines whether or not to plot the residual
"""
simplot
@recipe function simplot(p::Simplot; ploty = true, plote = false, sysname="")
    sys, d, x0 = _process_simplotargs(p.args...)
    y = time2(output(d))
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
        label --> [sysname*" "*"sim fit $(d.ny == 1 ? "" : i) :$(round(err, digits=2))%" for (i,err) in enumerate(err')]
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
	predplot(sys, data, x0=:estimate; ploty=true, plote=false, h=1, sysname="")

Plot system `h`-step prediction and measured output to compare them.

By default, the initial condition `x0` is estimated using the data. To start the simulation from the origin, provide `x0 = :zero` or `x0 = zeros(sys.nx)`.

- `ploty` determines whether or not to plot the measured signal
- `plote` determines whether or not to plot the residual
- `h` is the prediction horizon.
"""
predplot
@recipe function predplot(p::Predplot; ploty = true, plote = false, h=1, sysname="")
    sys, d = p.args[1:2]
    y = time2(output(d))
    u = time2(input(d))
    yh = predict(p.args...; h)
    xguide --> "Time [s]"
    yguide --> "Output"
    t = timevec(d)
    err = nrmse(y, yh)
    ploty && @series begin
        label --> ["y$i" for i in (1:d.ny)']
        t, y'
    end
    @series begin
        label --> [sysname*" "*(h > 1 ? "Horizon $h " : "")*"pred fit $(d.ny == 1 ? "" : i) :$(round(err, digits=2))%" for (i,err) in enumerate(err')]
        t, yh'
    end
    plote && @series begin
        label --> "pred resid."
        t, y' - yh'
    end
    nothing
end


@recipe function plot_frd(frd::FRD; hz = false, plotphase=false)
    xscale --> :log10
    xguide --> (hz ? "Frequency [Hz]" : "Frequency [rad/s]")
    label --> ""
    if ControlSystemsBase.issiso(frd)
        r = reshape(frd.r, 1, 1, :)
    else
        r = frd.r
    end
    ny,nu,nw = size(r)
    s2i(i,j) = LinearIndices((nu,(plotphase ? 2 : 1)*ny))[j,i]
    layout --> ((plotphase ? 2 : 1)*ny, nu)
    link --> :x
    for j=1:nu
        for i=1:ny
            @series begin
                subplot   --> min(s2i((plotphase ? (2i-1) : i),j), prod(plotattributes[:layout]))
                yguide --> "Magnitude"
                yscale --> :log10
                inds = findall(x -> x == 0, frd.w)
                useinds = setdiff(1:length(frd.w), inds)
                (hz ? 1 / (2π) : 1) .* frd.w[useinds], abs.(r[i,j,useinds])
            end
            if plotphase
                @series begin
                    yguide --> "Phase [deg]"
                    subplot   --> s2i(2i,j)
                    inds = findall(x -> x == 0, frd.w)
                    useinds = setdiff(1:length(frd.w), inds)
                    (hz ? 1 / (2π) : 1) .* frd.w[useinds], 180/pi .* ControlSystemsBase.unwrap(angle.(r[i,j,useinds]))
                end
            end
        end
    end
    nothing
end

# tfest returns a tuple and it's convenient to call plot(tfest(d)) directly
@recipe function plot_tfestres(frd::Tuple{<:FRD,<:FRD})
    @series begin
        label --> "Estimated TF"
        frd[1]
    end
    @series begin
        label --> "Noise"
        frd[2]
    end
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
    label --> false
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
    impulseestplot(data,n; σ = 2)

Estimates the system impulse response by fitting an `n`:th order FIR model and plots the result with a 95% (2σ) confidence band. Note, the confidence bound is drawn around zero, i.e., it is drawn such that one can determine whether or not the impulse response is significantly different from zero.

This method only supports single-output data, use [`okid`](@ref) for multi-output data.

See also [`impulseest`](@ref) and [`okid`](@ref).
"""
impulseestplot
@recipe function impulseestplot(p::Impulseestplot; λ = 0, σ = 2, cov=true)
    d = p.args[1]
    n = length(p.args) >= 2 ? p.args[2] : 25
    ir, t, Σ = impulseest(d, n; λ, cov)
    title --> "Estimated Impulse Response"
    xguide --> "Time [s]"

    seriestype --> :sticks
    @series begin
        label --> ""
        t, ir
    end
    linestyle := :dash
    seriescolor := :black
    label := ""
    seriestype := :line
    if cov 
        S = σ .* sqrt.(diag(Σ))
        if d.nu > 1
            S = reshape(S, :, d.nu)
        end
        @series begin
            t, S
        end
        @series begin
            t, -S
        end
    end
end

function _process_crosscor_args(d::AbstractIdData, lags = -min(length(d) ÷ 10, 100):min(length(d) ÷ 2, 100))
    time1(d.u), time1(d.y), lags, d.Ts
end

function _process_crosscor_args(u::AbstractArray, y::AbstractArray, Ts::Real, lags = -min(size(u,2) ÷ 10, 100):min(size(u,2) ÷ 2, 100))
    time1(u), time1(y), lags, Ts
end

function _process_autocor_args(y::AbstractArray, Ts::Real, lags = 1:min(size(y,2) ÷ 2, 100))
    time1(y), lags, Ts
end

@userplot Crosscorplot
@userplot Autocorplot

"""
    crosscorplot(data, [lags])
    crosscorplot(u, y, Ts, [lags])

Plot the cross correlation between input and output for `lags` that default to 10% of the length of the dataset on the negative side and 50% on the positive side but no more than 100 on each side.
"""
crosscorplot

"""
    autocorplot(y, Ts, [lags])

Plot the auto correlation of `y` for `lags` that default to `1:size(y, 2)÷2`.
"""
autocorplot

@recipe function crosscorplot(p::Crosscorplot)
    u,y,lags,Ts = _process_crosscor_args(p.args...)

    if size(u,2) == 1 && u isa AbstractMatrix
        u = vec(u)
    end
    if size(y,2) == 1 && y isa AbstractMatrix
        y = vec(y)
    end
    plotattributes[:N] = size(u, 1)
    xc = crosscor(u, y, lags, demean = true)
    title --> "Input-Output cross correlation"
    seriestype := :corrplot
    @series begin
        label --> ""
        lags .* Ts, xc
    end
end

@recipe function autocorplot(p::Autocorplot)
    y,lags,Ts = _process_autocor_args(p.args...)
    @show lags
    if size(y,2) == 1 && y isa AbstractMatrix
        y = vec(y)
    end
    plotattributes[:N] = size(y, 1)
    xc = autocor(y, lags, demean = true)
    title --> "Auto correlation"
    seriestype := :corrplot
    @series begin
        label --> ""
        lags .* Ts, xc
    end
end

@recipe function f(::Type{Val{:corrplot}}, plt::AbstractPlot)
    x, y = plotattributes[:x], plotattributes[:y]
    N = plotattributes[:N]
    seriestype := :sticks
    xguide --> "Lag [s]"
    @series begin
        x, y
    end
    
    # label := ""
    primary := false
    # Ni = N .- abs.(lags)
    linestyle := :dash
    seriescolor := :black
    framestyle --> :zerolines
    @series begin
        seriestype := :hline
        # lags.*d.Ts, 2 .*sqrt.(1 ./ Ni) # The denominator in crosscorr already takes care of this
        y := [2 .* sqrt.(1 ./ N)]
    end
    @series begin
        seriestype := :hline
        # lags.*d.Ts, -2 .*sqrt.(1 ./ Ni)
        y := [-2 .* sqrt.(1 ./ N)]
    end
end

@userplot Residualplot

function _process_res_args(sys, d::AbstractIdData, lags = -min(length(d) ÷ 10, 100):min(length(d) ÷ 2, 100))
    sys, d, lags
end

"""
    residualplot(model, data)

Plot residual autocorrelation and input-residual correlation.
"""
residualplot
@recipe function residualplot(p::Residualplot; h=1)
    sys, d, lags = _process_res_args(p.args...)
    lagsac = 1:maximum(lags)
    yh = isfinite(h) ? predict(sys, d; h) : simulate(sys, d)
    e = d.y - yh
    plotattributes[:N] = size(e, 2)
    xc = crosscor(time1(d.u), e', lags, demean = true)
    ac = autocor(e', lagsac, demean = true)
    xc = reshape(xc, size(xc, 1), :)
    layout := (2, 1)
    seriestype := :corrplot
    subplot := 1
    link --> :x
    @series begin
        title --> "Residual auto correlation"
        label --> ""
        lagsac .* d.Ts, ac
    end
    subplot := 2
    @series begin
        title --> "Input-residual cross correlation"
        label --> ""
        lags .* d.Ts, xc
    end
end


function ControlSystemsBase.gangoffour(P::FRD, C::FRD, ω = nothing)
    ω === nothing || ω == P.ω || error("Inconsistent frequency vectors")
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
    size(y, 1) == 1 || size(y, 2) == 1 || throw(ArgumentError("Only one-dimensional time series supported."))
    y = vec(y)
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
    find_nanb(d::InputOutputData, na, nb; logrms = false, method = :aic)
Plots the RMSE and AIC For model orders up to `na`, `nb`. Useful for model selection. `na` can be either an integer or a range. The same holds for `nb`.

- `logrms`: determines whether or not to plot the base 10 logarithm of the RMS error.
- `method`: determines whether to use the Akaike Information Criterion (`:aic`) or the Final Prediction Error (`:fpe`) to determine the model order.

If the color scale is hard to read due to a few tiles representing very large errors, avoid drawing those tiles by providing ranges for `na` and `nb` instead of integers, e.g., avoid showing model order smaller than 2 using `find_nanb(d, 3:na, 3:nb)`
"""
find_nanb
@recipe function find_nanb(p::Find_nanb; logrms = false, method = :aic)
    d, na, nb = p.args[1:3]
    d.ny == 1 || throw(ArgumentError("Only one-dimensional outputs supported."))
    y, u = vec(output(d)), time1(input(d))
    na_range = na isa Int ? (1:na) : na
    nb_range = nb isa Int ? (1:nb) : nb
    error = zeros(length(na_range), length(nb_range), 2)
    for (i, na) = enumerate(na_range), (j, nb) = enumerate(nb_range)
        yt, A = getARXregressor(y, u, na, nb)
        e = yt - A * (A \ yt)
        error[i, j, 1] = logrms ? log10.(rms(e)) : rms(e)
        error[i, j, 2] = method === :aic ? aic(e, na + nb) : fpe(e isa AbstractVector ? e : e', na + nb)
    end
    layout --> 2
    seriestype --> :heatmap
    xticks := (nb_range, nb_range)
    yticks := (na_range, na_range)
    yguide := "na"
    xguide := "nb"
    @series begin
        title := "RMS error"
        subplot := 1
        nb_range, na_range, error[:, :, 1]
    end
    @series begin
        title := (method === :aic ? "AIC" : "FPE")
        subplot := 2
        nb_range, na_range, error[:, :, 2]
    end
end