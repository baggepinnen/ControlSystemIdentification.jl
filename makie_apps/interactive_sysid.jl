"""
# Interactive System Identification App

This example demonstrates an interactive Makie app for system identification
with support for data loading, preprocessing, multiple identification methods,
and model comparison.

## Features:
1. Load data from CSV files
2. Preprocess data (detrend, remove offset, downsample)
3. Choose identification method (subspaceid or newpem) with options
4. Identify multiple models and compare them
5. Visualize prediction and simulation fits
6. Export identified models to workspace

## Usage:
```julia
using ControlSystemIdentification
include("makie_apps/interactive_sysid.jl")

# Launch the app
fig = interactive_sysid()
display(fig)

# Or with existing data
d = iddata(y, u, Ts)
fig = interactive_sysid(d)
display(fig)
```

## Requirements:
- ControlSystemIdentification
- GLMakie (for interactive features)
- CSV, DataFrames (for data loading)
"""

using ControlSystemIdentification
using ControlSystemIdentification: AbstractIdData, time2
using GLMakie
using Printf
using CSV
using DataFrames
using Statistics

# Data structure to store identified models
mutable struct IdentifiedModel
    name::String
    sys::Any  # AbstractPredictionStateSpace or similar
    method::String
    nx::Int
    options::Dict{Symbol, Any}
    fit_pred::Float64
    fit_sim::Float64
    color::RGBf
    visible::Observable{Bool}
end

"""
    load_csv_data(filepath::String, Ts::Real)

Load data from CSV file and create an iddata object.
Assumes CSV has columns for inputs (u1, u2, ...) and outputs (y1, y2, ...).
"""
function load_csv_data(filepath::String, Ts::Real)
    df = CSV.read(filepath, DataFrame)

    # Try to identify input and output columns
    # Convention: columns starting with 'u' are inputs, 'y' are outputs
    y_cols = filter(x -> startswith(string(x), "y"), names(df))
    u_cols = filter(x -> startswith(string(x), "u"), names(df))

    if isempty(y_cols) && isempty(u_cols)
        # If no naming convention, assume first half is outputs, second half is inputs
        ncols = size(df, 2)
        if ncols >= 2
            y_cols = names(df)[1:div(ncols, 2)]
            u_cols = names(df)[div(ncols, 2)+1:end]
        else
            error("Could not identify input/output columns in CSV")
        end
    end

    if isempty(y_cols)
        error("No output columns found")
    end

    y = Matrix(df[:, y_cols])'

    if !isempty(u_cols)
        u = Matrix(df[:, u_cols])'
        return iddata(y, u, Ts)
    else
        return iddata(y, Ts)
    end
end

"""
    preprocess_data(d::AbstractIdData; detrend_data=false, remove_offset=false, downsample_factor=1)

Apply preprocessing operations to iddata.
"""
function preprocess_data(d::AbstractIdData; detrend_data=false, remove_offset=false, downsample_factor=1)
    d_proc = d

    # Detrend (remove mean)
    if detrend_data
        d_proc = detrend(d_proc)
    end

    # Remove offset (subtract first value)
    if remove_offset
        y = time1(output(d_proc))
        y0 = y[1:1, :]
        y_offset = y .- y0

        u = time1(input(d_proc))
        u0 = u[1:1, :]
        u_offset = u .- u0
        d_proc = iddata(y_offset, u_offset, d_proc.Ts)
    end

    # Downsample
    if downsample_factor > 1
        f = 1.0 / downsample_factor
        d_proc = resample(d_proc, f)
    end

    return d_proc
end

"""
    identify_model(d::AbstractIdData, method::String, nx::Int, options::Dict)

Identify a model using the specified method and options.
Returns the identified system.
"""
function identify_model(d::AbstractIdData, method::String, nx::Int, options::Dict)
    if method == "subspaceid"
        sys = subspaceid(d, nx;
            zeroD = get(options, :zeroD, false),
            stable = get(options, :stable, true),
            focus = get(options, :focus, :prediction)
        )
    elseif method == "newpem"
        sys, _ = newpem(d, nx;
            zeroD = get(options, :zeroD, true),
            stable = get(options, :stable, true),
            focus = get(options, :focus, :prediction),
            h = get(options, :h, 1),
            show_trace = false,
            iterations = 100
        )
    else
        error("Unknown method: $method")
    end

    return sys
end

"""
    compute_fit(sys, d::AbstractIdData)

Compute prediction and simulation fit percentages for a model.
Returns (fit_pred, fit_sim).
"""
function compute_fit(sys, d::AbstractIdData)
    y = time2(output(d))

    # Prediction fit
    yh_pred = predict(sys, d)
    fit_pred = 100 * (1 - sqrt(sum((y - yh_pred).^2) / sum((y .- mean(y)).^2)))

    # Simulation fit
    yh_sim = simulate(sys, d)
    fit_sim = 100 * (1 - sqrt(sum((y - yh_sim).^2) / sum((y .- mean(y)).^2)))

    return (fit_pred, fit_sim)
end

"""
    interactive_sysid(initial_data=nothing)

Create an interactive system identification app.

# Arguments
- `initial_data`: Optional initial iddata to load

# Returns
- `fig`: The interactive Makie figure
"""
function interactive_sysid(initial_data=nothing)
    # Observables for data and models
    current_data = Observable{Any}(initial_data)
    processed_data = Observable{Any}(initial_data)
    model_list = Observable{Vector{IdentifiedModel}}([])
    next_model_id = Ref(1)

    # UI state observables
    csv_path = Observable("")
    sample_time_str = Observable("0.1")
    detrend_toggle = Observable(false)
    remove_offset_toggle = Observable(false)
    downsample_factor = Observable(1)

    # Identification options
    method_choice = Observable("subspaceid")
    model_order_str = Observable("5")
    zeroD_toggle = Observable(false)
    stable_toggle = Observable(true)
    focus_choice = Observable(:prediction)
    h_value_str = Observable("1")

    # Status message
    status_msg = Observable("Load data or select initial data to begin")

    # Available color palette for models
    color_palette = [
        colorant"#1f77b4", colorant"#ff7f0e", colorant"#2ca02c",
        colorant"#d62728", colorant"#9467bd", colorant"#8c564b",
        colorant"#e377c2", colorant"#7f7f7f", colorant"#bcbd22",
        colorant"#17becf"
    ]

    # Create figure
    fig = Figure(size=(1400, 1000))

    # ===== DATA LOADING SECTION =====
    Label(fig[1, 1:4], "System Identification App", fontsize=20, font=:bold)

    # CSV loading
    Label(fig[2, 1], "CSV Path:", halign=:right)
    csv_textbox = Textbox(fig[2, 2:3], placeholder="path/to/data.csv")
    on(csv_textbox.stored_string) do val
        csv_path[] = something(val, "")
    end

    Label(fig[2, 4], "Ts:", halign=:right)
    ts_textbox = Textbox(fig[2, 5], placeholder="0.1",
                         validator=x -> tryparse(Float64, x) !== nothing)
    on(ts_textbox.stored_string) do val
        sample_time_str[] = something(val, "0.1")
    end

    load_btn = Button(fig[2, 6], label="Load CSV")

    # Preprocessing controls
    Label(fig[3, 1], "Preprocessing:", halign=:right)
    detrend_chk = Toggle(fig[3, 2], active=detrend_toggle)
    Label(fig[3, 2], "Detrend", halign=:left, padding=(80, 0, 0, 0))

    offset_chk = Toggle(fig[3, 3], active=remove_offset_toggle)
    Label(fig[3, 3], "Remove Offset", halign=:left, padding=(80, 0, 0, 0))

    Label(fig[3, 4], "Downsample:", halign=:right)
    downsample_slider = Slider(fig[3, 5], range=1:10, startvalue=1)
    connect!(downsample_factor, downsample_slider.value)
    Label(fig[3, 6], @lift(string($downsample_factor)), halign=:left)

    apply_preproc_btn = Button(fig[3, 7], label="Apply")

    # Data info display
    data_info = Observable("No data loaded")
    Label(fig[4, 1:4], data_info, halign=:left)

    # ===== DATA PREVIEW PLOT =====
    data_preview_ax = Axis(fig[5, 1:7],
                           xlabel="Time [s]",
                           ylabel="Signals",
                           title="Data Preview")

    # ===== IDENTIFICATION OPTIONS SECTION =====
    Label(fig[6, 1:2], "Identification Options", fontsize=16, font=:bold)

    Label(fig[7, 1], "Method:", halign=:right)
    method_menu = Menu(fig[7, 2], options=["subspaceid", "newpem"], default="subspaceid")
    connect!(method_choice, method_menu.selection)

    Label(fig[7, 3], "Order (nx):", halign=:right)
    order_textbox = Textbox(fig[7, 4], placeholder="5",
                           validator=x -> tryparse(Int, x) !== nothing)
    on(order_textbox.stored_string) do val
        model_order_str[] = something(val, "5")
    end

    # Method-specific options
    Label(fig[8, 1], "zeroD:", halign=:right)
    zeroD_chk = Toggle(fig[8, 2], active=zeroD_toggle)

    Label(fig[8, 3], "stable:", halign=:right)
    stable_chk = Toggle(fig[8, 4], active=stable_toggle)

    Label(fig[9, 1], "focus:", halign=:right)
    focus_menu = Menu(fig[9, 2], options=["prediction", "simulation"], default="prediction")
    on(focus_menu.selection) do val
        focus_choice[] = Symbol(val)
    end

    # h parameter for newpem (shown always, only used for newpem)
    h_label = Label(fig[9, 3], "h:", halign=:right)
    h_textbox = Textbox(fig[9, 4], placeholder="1",
                       validator=x -> tryparse(Int, x) !== nothing)
    on(h_textbox.stored_string) do val
        h_value_str[] = something(val, "1")
    end
    Label(fig[9, 5], "(newpem only)", fontsize=10, halign=:left)

    identify_btn = Button(fig[10, 1:2], label="Identify Model", buttoncolor=:green)

    # ===== MODEL LIST SECTION =====
    Label(fig[6, 4:7], "Identified Models", fontsize=16, font=:bold)

    # Model list display - will be updated dynamically with buttons
    model_list_scroll = GridLayout(fig[7:10, 4:7], tellheight=false)

    # Function to rebuild model list UI
    function rebuild_model_list_ui!()
        # Clear existing content
        for i in length(contents(model_list_scroll)):-1:1
            delete!(contents(model_list_scroll)[i])
        end

        models = model_list[]
        if isempty(models)
            Label(model_list_scroll[1, 1:3], "No models yet", fontsize=12)
        else
            for (i, model) in enumerate(models)
                # Model info
                info_text = "$(i). $(model.name) ($(model.method), nx=$(model.nx))\n   Pred: $(round(model.fit_pred, digits=1))%  Sim: $(round(model.fit_sim, digits=1))%"
                Label(model_list_scroll[i, 1], info_text, fontsize=11, halign=:left,
                      justification=:left, tellwidth=false)

                # Remove button for this model
                remove_btn = Button(model_list_scroll[i, 2], label="×",
                                   width=30, height=30)
                # Capture the model in the closure, not the index
                let model = model
                    on(remove_btn.clicks) do _
                        # Find and remove this specific model from the list
                        idx = findfirst(m -> m === model, model_list[])
                        if idx !== nothing
                            deleteat!(model_list[], idx)
                            notify(model_list)

                            # Update plots (rebuild_model_list_ui! is called by on(model_list))
                            if processed_data[] !== nothing && !isempty(model_list[])
                                update_plots!(pred_ax, sim_ax, model_list[], processed_data[],
                                             pred_legend, sim_legend)
                            elseif isempty(model_list[])
                                # Clear plots if no models left
                                empty!(pred_ax)
                                empty!(sim_ax)
                            end

                            status_msg[] = "$(model.name) removed"
                        end
                    end
                end
            end
        end
    end

    # Update model list UI when models change
    on(model_list) do _
        rebuild_model_list_ui!()
    end

    # Initialize
    rebuild_model_list_ui!()

    clear_all_btn = Button(fig[11, 4:5], label="Clear All")
    export_btn = Button(fig[11, 6:7], label="Export to Workspace")

    # ===== VISUALIZATION SECTION =====
    # Prediction plot
    pred_ax = Axis(fig[12, 1:7],
                   xlabel="Time [s]",
                   ylabel="Output",
                   title="Prediction Plot")

    # Simulation plot
    sim_ax = Axis(fig[13, 1:7],
                  xlabel="Time [s]",
                  ylabel="Output",
                  title="Simulation Plot")

    # Link x-axes
    linkyaxes!(pred_ax, sim_ax)

    # Store legend references so we can delete them later
    pred_legend = Ref{Union{Nothing, Legend}}(nothing)
    sim_legend = Ref{Union{Nothing, Legend}}(nothing)

    # Status bar
    Label(fig[14, 1:7], status_msg, halign=:left, color=:blue)

    # ===== CALLBACKS =====

    # Load CSV callback
    on(load_btn.clicks) do _
        try
            path = csv_path[]
            Ts_parsed = tryparse(Float64, sample_time_str[])

            if isempty(path)
                status_msg[] = "Error: Please provide a CSV path"
                return
            end

            if Ts_parsed === nothing
                status_msg[] = "Error: Invalid sample time"
                return
            end

            d = load_csv_data(path, Ts_parsed)
            current_data[] = d
            processed_data[] = d

            data_info[] = "Loaded: $(noutputs(d)) outputs, $(ninputs(d)) inputs, $(length(d)) samples, Ts=$(d.Ts)s"
            status_msg[] = "Data loaded successfully from $path"

            # Update preview plot
            update_data_preview!(data_preview_ax, d)
        catch e
            status_msg[] = "Error loading CSV: $(sprint(showerror, e))"
        end
    end

    # Apply preprocessing callback
    on(apply_preproc_btn.clicks) do _
        if current_data[] === nothing
            status_msg[] = "Error: No data loaded"
            return
        end

        try
            d = current_data[]
            d_proc = preprocess_data(d;
                detrend_data=detrend_toggle[],
                remove_offset=remove_offset_toggle[],
                downsample_factor=downsample_factor[])

            processed_data[] = d_proc
            status_msg[] = "Preprocessing applied"

            # Update preview plot
            update_data_preview!(data_preview_ax, d_proc)
        catch e
            status_msg[] = "Error in preprocessing: $(sprint(showerror, e))"
        end
    end

    # Identify model callback
    on(identify_btn.clicks) do _
        if processed_data[] === nothing
            status_msg[] = "Error: No data loaded"
            return
        end

        try
            status_msg[] = "Identifying model... (this may take a moment)"

            d = processed_data[]
            method = method_choice[]

            nx_parsed = tryparse(Int, model_order_str[])
            if nx_parsed === nothing
                status_msg[] = "Error: Invalid model order"
                return
            end
            nx = nx_parsed

            options = Dict{Symbol, Any}(
                :zeroD => zeroD_toggle[],
                :stable => stable_toggle[],
                :focus => focus_choice[]
            )

            if method == "newpem"
                h_parsed = tryparse(Int, h_value_str[])
                if h_parsed === nothing
                    status_msg[] = "Error: Invalid h value"
                    return
                end
                options[:h] = h_parsed
            end

            # Identify the system
            sys = identify_model(d, method, nx, options)

            # Compute fits
            fit_pred, fit_sim = compute_fit(sys, d)

            # Create model entry
            model_name = "Model $(next_model_id[])"
            next_model_id[] += 1

            color_idx = length(model_list[]) % length(color_palette) + 1

            model = IdentifiedModel(
                model_name,
                sys,
                method,
                nx,
                options,
                fit_pred,
                fit_sim,
                color_palette[color_idx],
                Observable(true)
            )

            # Add to list
            push!(model_list[], model)
            notify(model_list)

            status_msg[] = "$model_name identified: pred=$(round(fit_pred, digits=1))%, sim=$(round(fit_sim, digits=1))%"

            # Update plots
            update_plots!(pred_ax, sim_ax, model_list[], processed_data[], pred_legend, sim_legend)

        catch e
            rethrow()
            status_msg[] = "Error identifying model: $(sprint(showerror, e))"
        end
    end

    # Clear all models callback
    on(clear_all_btn.clicks) do _
        empty!(model_list[])
        notify(model_list)
        next_model_id[] = 1
        status_msg[] = "All models cleared"

        # Clear plots
        empty!(pred_ax)
        empty!(sim_ax)
    end

    # Export models callback
    on(export_btn.clicks) do _
        if isempty(model_list[])
            status_msg[] = "No models to export"
            return
        end

        # Export visible models to Main module
        exported = 0
        for model in model_list[]
            if model.visible[]
                varname = Symbol(replace(model.name, " " => "_"))
                Core.eval(Main, :($varname = $(model.sys)))
                exported += 1
            end
        end

        status_msg[] = "Exported $exported model(s) to workspace"
    end

    # Note: Model list UI updates are handled by rebuild_model_list_ui!()
    # which is called from the on(model_list) callback defined earlier

    # Initialize with initial data if provided
    if initial_data !== nothing
        d = initial_data
        data_info[] = "Initial: $(noutputs(d)) outputs, $(ninputs(d)) inputs, $(length(d)) samples, Ts=$(d.Ts)s"
        update_data_preview!(data_preview_ax, d)
    end

    return fig
end

"""
    update_data_preview!(ax, d::AbstractIdData)

Update the data preview plot with current data.
"""
function update_data_preview!(ax, d::AbstractIdData)
    empty!(ax)

    t = timevec(d)
    y = time2(output(d))

    # Plot outputs
    for i in 1:noutputs(d)
        lines!(ax, t, y[i, :], label="y$i")
    end

    # Plot inputs if available
    u = time2(input(d))
    for i in 1:ninputs(d)
        lines!(ax, t, u[i, :], label="u$i", linestyle=:dash)
    end

    axislegend(ax, position=:rt)
end

"""
    update_plots!(pred_ax, sim_ax, models, d::AbstractIdData, pred_legend, sim_legend)

Update prediction and simulation plots with all visible models.
"""
function update_plots!(pred_ax, sim_ax, models, d::AbstractIdData, pred_legend, sim_legend)
    # Delete old legends if they exist
    if pred_legend[] !== nothing
        delete!(pred_legend[])
        pred_legend[] = nothing
    end
    if sim_legend[] !== nothing
        delete!(sim_legend[])
        sim_legend[] = nothing
    end

    # Clear axes
    empty!(pred_ax)
    empty!(sim_ax)

    t = timevec(d)
    y = time2(output(d))

    # Plot measured output
    ny = noutputs(d)
    for i in 1:ny
        lines!(pred_ax, t, y[i, :], color=:black, linewidth=2, label="y$i measured")
        lines!(sim_ax, t, y[i, :], color=:black, linewidth=2, label="y$i measured")
    end

    # Plot each visible model
    for model in models
        if !model.visible[]
            continue
        end

        try
            # Prediction
            yh_pred = predict(model.sys, d)
            for i in 1:ny
                label = "$(model.name) pred ($(round(model.fit_pred, digits=1))%)"
                lines!(pred_ax, t, yh_pred[i, :], color=model.color, linewidth=1.5,
                      label=(i == 1 ? label : ""))
            end

            # Simulation
            yh_sim = simulate(model.sys, d)
            for i in 1:ny
                label = "$(model.name) sim ($(round(model.fit_sim, digits=1))%)"
                lines!(sim_ax, t, yh_sim[i, :], color=model.color, linewidth=1.5,
                      label=(i == 1 ? label : ""))
            end
        catch e
            rethrow()
            println("Warning: Could not plot model $(model.name): $e")
        end
    end

    # Create new legends and store references
    if !isempty(pred_ax.scene.plots)
        pred_legend[] = axislegend(pred_ax, position=:rb, merge=true, unique=true)
    end
    if !isempty(sim_ax.scene.plots)
        sim_legend[] = axislegend(sim_ax, position=:rb, merge=true, unique=true)
    end
end

"""
    demo_interactive_sysid()

Launch the interactive system identification app with demo data.
"""
function demo_interactive_sysid()
    println("Interactive System Identification App")
    println("=====================================")

    # Create demo data
    @eval using ControlSystemsBase

    # True system
    G = tf(1, [1, 0.5, 1]) * tf([1, 0.1], [1, 0.8])
    Gd = c2d(G, 0.1)

    # Generate data
    T = 500
    u = randn(1, T)  # 1×T matrix, time in second dimension
    y, t_sim, _ = lsim(Gd, u)  # Returns T×1, need to transpose
    y = Matrix(y')  # Transpose to 1×T
    y = y .+ 0.1 .* randn(size(y))  # Add noise

    d = iddata(y, u, 0.1)

    println("\nDemo data generated:")
    println("  - System order: 4")
    println("  - Samples: $T")
    println("  - Sample time: 0.1s")
    println("  - 1 input, 1 output")

    fig = interactive_sysid(d)
    display(fig)

    println("\nApp launched! Try:")
    println("  1. Adjust preprocessing options and click 'Apply'")
    println("  2. Choose method (subspaceid or newpem)")
    println("  3. Set model order (try nx=4 for this demo)")
    println("  4. Click 'Identify Model'")
    println("  5. Compare multiple models")
    println("  6. Export models to workspace")

    return fig
end

# Allow running as a script
demo_interactive_sysid()