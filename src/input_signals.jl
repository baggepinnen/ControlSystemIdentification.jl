"""
    InputSignals

A submodule for generating standard input signals commonly used in system identification.
This module provides functions for generating various types of excitation signals including
Pseudo-Random Binary Sequences (PRBS), chirp signals, multi-sine signals, and step signals.
"""
module InputSignals

using Random
using LinearAlgebra

export prbs, chirp, multisine, step_signal

"""
    prbs(N; low=-1, high=1, seed=nothing, period=1)

Generate a Pseudo-Random Binary Sequence (PRBS) of length `N`.

A PRBS is a deterministic signal that approximates white noise properties while taking
only two amplitude values. It's widely used in system identification due to its
flat power spectrum and ease of generation.

# Arguments
- `N::Int`: Length of the sequence
- `low::Real=-1`: Lower amplitude value (default: -1)
- `high::Real=1`: Upper amplitude value (default: 1)
- `seed::Union{Int,Nothing}=nothing`: Random seed for reproducibility
- `period::Int=1`: Minimum number of samples before the signal can change (dominant period)

# Returns
- `Vector{Float64}`: PRBS signal of length N with values alternating between `low` and `high`

# Examples
```julia
# Generate a standard PRBS with values {-1, 1}
u = prbs(100)

# Generate a PRBS with custom amplitude levels
u = prbs(100; low=0, high=5)

# Generate a reproducible PRBS
u = prbs(100; seed=42)

# Generate a slower PRBS that holds values for at least 5 samples
u = prbs(100; period=5)
```

# Notes
- The algorithm uses the sign of normally distributed random numbers to generate
  the binary sequence. While not a true maximal-length sequence, it provides
  good statistical properties for system identification purposes.
- The `period` parameter controls the spectral content by setting a minimum hold time,
  effectively creating a low-pass characteristic in the signal spectrum.
"""
function prbs(N::Int; low::Real=-1, high::Real=1, seed::Union{Int,Nothing}=nothing, period::Int=1)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    period >= 1 || throw(ArgumentError("period must be >= 1"))
    
    if period == 1
        # Standard PRBS - can change every sample
        sequence = sign.(randn(N))
    else
        # Generate base sequence with appropriate length
        base_length = ceil(Int, N / period)
        base_sequence = sign.(randn(base_length))
        
        # Repeat each value 'period' times
        expanded_sequence = repeat(base_sequence, inner=period)
        
        # Truncate to exact length N
        sequence = expanded_sequence[1:N]
    end
    
    # Map from {-1, 1} to {low, high}
    if low != -1 || high != 1
        # Linear mapping: -1 -> low, 1 -> high
        @. sequence = (sequence + 1) / 2 * (high - low) + low
    end
    
    return sequence
end

"""
    chirp(N; Ts, f0, f1, logspace=true)

Generate a chirp (frequency sweep) signal of length `N`.

A chirp signal is a sinusoidal signal whose frequency changes over time.
It's useful for system identification as it excites the system across
a range of frequencies.

# Arguments
- `N::Int`: Length of the signal
- `Ts::Real`: Sample time
- `f0::Real`: Starting frequency (Hz)
- `f1::Real`: Ending frequency (Hz)
- `logspace::Bool=true`: If true, use logarithmic frequency spacing; if false, use linear

# Returns
- `Vector{Float64}`: Chirp signal

# Examples
```julia
# Generate a logarithmic chirp from 0.1 to 10 Hz
u = chirp(1000, Ts=0.01, f0=0.1, f1=10.0)

# Generate a linear chirp
u = chirp(1000, Ts=0.01, f0=0.1, f1=10.0, logspace=false)
```

# Notes
- Logarithmic spacing is often preferred for system identification as it provides
  equal energy per frequency decade
- The signal amplitude is normalized to ±1
"""
function chirp(N::Int; Ts::Real, f0::Real, f1::Real, logspace::Bool=true)
    if N == 0
        return Float64[]
    elseif N == 1
        # For single sample, use the starting frequency at t=0
        return [sin(2π * f0 * 0)]
    end
    
    t = range(0, step=Ts, length=N)
    T = t[end] - t[1]

    f0 > 0 && f1 > 0 || throw(ArgumentError("f0 and f1 must be positive frequencies"))
    
    if logspace
        f = f0 .* (f1/f0) .^ (t ./ T)
        phase = (2π * T / log(f1/f0)) .* (f .- f0)
    else
        f = LinRange(f0, f1, N)
        k = (f1 - f0) / T
        phase = 2π .* (f0 .* t .+ (k/2) .* t.^2)
    end
    
    q = sin.(phase)

    return q
end

"""
    multisine(N; frequencies, Ts)

Generate a multi-sine signal by summing sinusoidal components at specified frequencies.

Multi-sine signals are useful for system identification when you want to excite
the system at specific frequencies while avoiding others (e.g., to avoid resonances).

# Arguments
- `N::Int`: Length of the signal
- `frequencies::AbstractVector{<:Real}`: Vector of frequencies (Hz) to include
- `Ts::Real`: Sample time

# Returns
- `Vector{Float64}`: Multi-sine signal

# Examples
```julia
# Generate a multi-sine with frequencies at 0.1, 1.0, and 10.0 Hz
freqs = [0.1, 1.0, 10.0]
u = multisine(1000, frequencies=freqs, Ts=0.01)
```

# Notes
- The amplitude of each frequency component is normalized so that the total RMS
  value is approximately 1
- Phase of each component is randomized to minimize peak factor
"""
function multisine(N::Int; frequencies::AbstractVector{<:Real}, Ts::Real)
    if N == 0
        return Float64[]
    end
    
    if isempty(frequencies)
        return zeros(N)
    end
    
    t = range(0, step=Ts, length=N)
    
    # Generate random phases to minimize peak factor
    phases = 2π * rand(length(frequencies))
    
    # Sum sinusoidal components
    signal = zeros(N)
    for (i, freq) in enumerate(frequencies)
        @. signal += sin(2π * freq * t + phases[i])
    end
    
    # Normalize amplitude by sqrt of number of components
    signal ./= sqrt(length(frequencies))
    
    return signal
end

"""
    step_signal(N; step_time, Ts=1.0, offset=0.0, amplitude=1.0)

Generate a step signal that transitions from `offset` to `offset + amplitude` at the specified time.

Step signals are fundamental for system identification, particularly for
determining system response characteristics and time constants.

# Arguments
- `N::Int`: Length of the signal
- `step_time::Real`: Time at which the step occurs (in seconds)
- `Ts::Real=1.0`: Sample time (seconds)
- `offset::Real=0.0`: Initial value of the signal before the step
- `amplitude::Real=1.0`: Height of the step

# Returns
- `Vector{Float64}`: Step signal

# Examples
```julia
# Generate a step signal that steps at 5.0 seconds with Ts=0.1
u = step_signal(1000, step_time=5.0, Ts=0.1)

# Generate a step signal with custom offset and amplitude
u = step_signal(1000, step_time=2.5, Ts=0.1, offset=2.0, amplitude=3.0)
```

# Notes
- The signal is `offset` before `step_time` and `offset + amplitude` after `step_time`
- If step_time ≤ 0, the entire signal is `offset + amplitude`
- If step_time is beyond the signal duration, the entire signal is `offset`
"""
function step_signal(N::Int; step_time::Real, Ts::Real=1.0, offset::Real=0.0, amplitude::Real=1.0)
    if N == 0
        return Float64[]
    end
    
    step_index = round(Int, step_time / Ts) + 1
    signal = fill(offset, N)
    if step_index <= N && step_index >= 1
        signal[step_index:end] .= offset + amplitude
    elseif step_index < 1
        signal .= offset + amplitude
    end
    return signal
end

end # module InputSignals