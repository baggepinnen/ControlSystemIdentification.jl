# Input Signal Generation

This page describes the input signal generation capabilities of ControlSystemIdentification.jl. The `InputSignals` submodule provides functions for generating various types of excitation signals commonly used in system identification.

## Overview

Proper excitation signal design is crucial for successful system identification. The quality of the input signal directly affects the quality of the identified model. This package provides several standard signal types:

- **PRBS (Pseudo-Random Binary Sequence)**: Binary signals with white noise-like properties
- **Chirp signals**: Frequency sweeps for analyzing system response across frequency ranges
- **Multi-sine signals**: Controlled spectral content at specific frequencies
- **Step signals**: Fundamental for transient response analysis

## PRBS (Pseudo-Random Binary Sequence)

```@docs
prbs
```

### Example

```@example input_signals
using ControlSystemIdentification, Plots

# Generate a standard PRBS signal
N = 200
u_prbs = prbs(N; seed=42)

# Generate a PRBS with custom amplitude levels
u_custom = prbs(N; low=0, high=5, seed=42)

# Plot the signals
p1 = plot(u_prbs, title="Standard PRBS", xlabel="Sample", ylabel="Amplitude", 
          label="u(t)", linewidth=2)
p2 = plot(u_custom, title="Custom PRBS (0 to 5)", xlabel="Sample", ylabel="Amplitude", 
          label="u(t)", linewidth=2)

plot(p1, p2, layout=(2,1), size=(600, 400))
```

## Chirp Signals

```@docs
chirp
```

### Example

```@example input_signals
# Generate chirp signals
N = 1000
Ts = 0.01
f0 = 1.0  # Start frequency (Hz)
f1 = 10.0 # End frequency (Hz)

u_log = chirp(N; Ts, f0, f1, logspace=true)
u_lin = chirp(N; Ts, f0, f1, logspace=false)

# Time vector for plotting
t = (0:N-1) * Ts

# Plot time domain signals
p1 = plot(t, u_log, title="Logarithmic Chirp", xlabel="Time (s)", ylabel="Amplitude", 
          label="u(t)", linewidth=1)
p2 = plot(t, u_lin, title="Linear Chirp", xlabel="Time (s)", ylabel="Amplitude", 
          label="u(t)", linewidth=1)

plot(p1, p2, layout=(2,1), size=(600, 400))
```

### Frequency Content Analysis

```@example input_signals
using ControlSystemIdentification.FFTW

# Analyze frequency content of the chirp signals
U_log = abs.(fft(u_log))
U_lin = abs.(fft(u_lin))

# Frequency axis (only plot positive frequencies)
freqs = (0:N-1) / (N*Ts)
half_N = N÷2

# Plot frequency domain (first half only, up to Nyquist frequency)
p1 = plot(freqs[2:half_N], U_log[2:half_N], title="Logarithmic Chirp Spectrum", 
          xlabel="Frequency (Hz)", ylabel="Magnitude", label="Log chirp", 
          xscale=:log10, yscale=:log10, linewidth=2)
p2 = plot(freqs[2:half_N], U_lin[2:half_N], title="Linear Chirp Spectrum", 
          xlabel="Frequency (Hz)", ylabel="Magnitude", label="Linear chirp", 
          xscale=:log10, yscale=:log10, linewidth=2)
vline!(p1, [f0, f1], label="", linestyle=:dash, color=:red, alpha=0.7)
vline!(p2, [f0, f1], label="", linestyle=:dash, color=:red, alpha=0.7)

plot(p1, p2, layout=(2,1), size=(600, 400))
```

Chirp signals are excellent for:
- Frequency response estimation
- Analyzing system behavior across frequency ranges
- Identifying resonances and anti-resonances
- Logarithmic chirps provide equal energy per frequency decade

## Multi-sine Signals

```@docs
multisine
```

### Example

```@example input_signals
# Generate multi-sine signal
N = 1000
Ts = 0.01
frequencies = [0.5, 2.0, 5.0, 10.0]  # Hz

u_multi = multisine(N; frequencies=frequencies, Ts)

# Time vector
t = (0:N-1) * Ts

# Plot time domain signal
p1 = plot(t, u_multi, title="Multi-sine Signal", xlabel="Time (s)", ylabel="Amplitude", 
          label="u(t)", linewidth=1)

# Analyze frequency content
U_multi = abs.(fft(u_multi))
freqs = (0:N-1) / (N*Ts)
half_N = N÷2

# Plot frequency domain
p2 = plot(freqs[1:half_N], U_multi[1:half_N], title="Multi-sine Spectrum", 
          xlabel="Frequency (Hz)", ylabel="Magnitude", label="Spectrum", 
          linewidth=2)

# Mark the target frequencies with vertical lines
for f in frequencies
    if f <= freqs[half_N]
        plot!(p2, [f, f], [0, maximum(U_multi[1:half_N])], 
              label="", linestyle=:dash, color=:red, alpha=0.7)
    end
end

plot(p1, p2, layout=(2,1), size=(600, 400))
```

Multi-sine signals are ideal for:
- Avoiding problematic frequencies (resonances)
- Controlled spectral content
- Nonlinear system identification
- Minimizing peak factor (crest factor)

## Step Signals

```@docs
step_signal
```

### Example

```@example input_signals
# Generate step signals
N = 200
Ts = 0.1
u_step1 = step_signal(N; step_time=5.0, Ts)   # Step at 5.0 seconds
u_step2 = step_signal(N; step_time=10.0, Ts)  # Step at 10.0 seconds

# Plot the signals
p1 = plot(u_step1, title="Step at 5.0 seconds", xlabel="Sample", ylabel="Amplitude", 
          label="u(t)", linewidth=2)
p2 = plot(u_step2, title="Step at 10.0 seconds", xlabel="Sample", ylabel="Amplitude", 
          label="u(t)", linewidth=2)

plot(p1, p2, layout=(2,1), size=(600, 400))
```

## Design Guidelines Video

```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/QMO8cDpjw5U?si=MZuC1BwKoLtgGJ_y" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```