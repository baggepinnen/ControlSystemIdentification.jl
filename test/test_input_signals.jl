using ControlSystemIdentification
using Test
using Random
using LinearAlgebra
using Statistics
using FFTW
import DSP
using DSP: xcorr

@testset "InputSignals Tests" begin
    
    @testset "PRBS Tests" begin
        # Test basic PRBS generation
        N = 100
        u = prbs(N)
        
        @test length(u) == N
        @test eltype(u) == Float64
        @test all(x -> x ∈ [-1, 1], u)  # Should only contain -1 and 1
        
        # Test custom amplitude levels
        u_custom = prbs(N, low=0, high=5)
        @test all(x -> x ∈ [0, 5], u_custom)
        
        # Test reproducibility with seed
        u1 = prbs(N, seed=42)
        u2 = prbs(N, seed=42)
        @test u1 == u2
        
        # Test different seeds produce different sequences
        u3 = prbs(N, seed=123)
        @test u1 != u3
        
        # Test that signal has approximately equal positive and negative values
        u_large = prbs(10000)
        pos_count = sum(u_large .> 0)
        @test abs(pos_count - 5000) < 500  # Should be approximately 50%
        
        # Test that the signal has good spectral properties (roughly flat spectrum)
        u_spec = prbs(1024)
        U = abs.(fft(u_spec))
        # Low frequency content should be significant
        @test U[2] > 0.1 * maximum(U)
        # High frequency content should also be significant
        @test U[end÷2] > 0.1 * maximum(U)
        
        # Test period parameter
        u_period = prbs(100, period=5, seed=42)
        @test length(u_period) == 100
        # Check that values are held for at least the specified period
        changes = sum(diff(u_period) .!= 0)
        max_possible_changes = floor(Int, 100 / 5)
        @test changes <= max_possible_changes
        
        # Test that period=1 gives same result as default
        u_default = prbs(100, seed=42)
        u_period1 = prbs(100, period=1, seed=42)
        @test u_default == u_period1
    end
    
    @testset "Chirp Tests" begin
        N = 1000
        Ts = 0.01
        f0 = 0.1
        f1 = 10.0
        
        # Test basic chirp generation
        u = chirp(N, Ts=Ts, f0=f0, f1=f1)
        @test length(u) == N
        @test eltype(u) == Float64
        
        # Test amplitude is normalized
        @test maximum(abs.(u)) <= 1.0
        @test maximum(abs.(u)) > 0.9  # Should be close to 1
        
        # Test linear vs logarithmic spacing
        u_log = chirp(N, Ts=Ts, f0=f0, f1=f1, logspace=true)
        u_lin = chirp(N, Ts=Ts, f0=f0, f1=f1, logspace=false)
        @test u_log != u_lin
        
        # Test frequency content - should have energy across the frequency range
        U_log = abs.(fft(u_log))
        U_lin = abs.(fft(u_lin))
        
        # Both should have significant energy distributed (not concentrated in DC)
        @test sum(U_log[2:end]) > 0.5 * sum(U_log)  # Most energy not in DC
        @test sum(U_lin[2:end]) > 0.5 * sum(U_lin)  # Most energy not in DC
        
        # Test edge cases
        u_single = chirp(1, Ts=Ts, f0=f0, f1=f1)
        @test length(u_single) == 1
        @test isfinite(u_single[1])
    end
    
    @testset "Multisine Tests" begin
        N = 1000
        Ts = 0.01
        frequencies = [0.1, 1.0, 10.0]
        
        # Test basic multisine generation
        u = multisine(N, frequencies=frequencies, Ts=Ts)
        @test length(u) == N
        @test eltype(u) == Float64
        
        # Test amplitude normalization (RMS should be approximately 1/sqrt(length))
        rms_value = sqrt(mean(u.^2))
        expected_rms = 1.0 / sqrt(2)  # For normalized sinusoids
        @test 0.5 < rms_value < 1.5  # Relaxed bounds for multiple sine components
        
        # Test frequency content
        U = abs.(fft(u))
        freq_axis = (0:N-1) / (N*Ts)
        
        # Check that energy is concentrated at the specified frequencies
        for target_freq in frequencies
            # Find the closest frequency bin
            _, idx = findmin(abs.(freq_axis .- target_freq))
            # Check that there's significant energy at this frequency
            @test U[idx] > 0.1 * maximum(U)
        end
        
        # Test with single frequency
        u_single = multisine(N, frequencies=[1.0], Ts=Ts)
        @test length(u_single) == N
        
        # Test with empty frequency vector
        u_empty = multisine(N, frequencies=Float64[], Ts=Ts)
        @test all(u_empty .== 0)
    end
    
    @testset "Step Signal Tests" begin
        N = 100
        Ts = 0.1
        
        # Test basic step signal
        step_time = 5.0  # 5 seconds
        u = step_signal(N, step_time=step_time, Ts=Ts)
        @test length(u) == N
        @test eltype(u) == Float64
        
        # Test step behavior (step at sample 51: 5.0/0.1 + 1 = 51)
        step_sample = round(Int, step_time / Ts) + 1
        @test all(u[1:step_sample-1] .== 0)
        @test all(u[step_sample:end] .== 1)
        
        # Test with custom offset and amplitude
        u_custom = step_signal(N, step_time=2.0, Ts=Ts, offset=2.0, amplitude=3.0)
        step_sample_custom = round(Int, 2.0 / Ts) + 1
        @test all(u_custom[1:step_sample_custom-1] .== 2.0)
        @test all(u_custom[step_sample_custom:end] .== 5.0)
        
        # Test edge cases
        u_early = step_signal(N, step_time=0.0, Ts=Ts)
        @test all(u_early .== 1)
        
        u_late = step_signal(N, step_time=20.0, Ts=Ts)  # Beyond signal duration
        @test all(u_late .== 0)
        
        # Test negative step time
        u_neg = step_signal(N, step_time=-1.0, Ts=Ts)
        @test all(u_neg .== 1)
    end
    
    @testset "Function Parameter Validation" begin
        # Test that functions handle edge cases properly
        
        # Test with N = 1
        @test length(prbs(1)) == 1
        @test length(chirp(1, Ts=0.01, f0=0.1, f1=1.0)) == 1
        @test length(multisine(1, frequencies=[1.0], Ts=0.01)) == 1
        @test length(step_signal(1, step_time=0.5, Ts=0.1)) == 1
        
        # Test with N = 0 (should handle gracefully)
        @test length(prbs(0)) == 0
        @test length(chirp(0, Ts=0.01, f0=0.1, f1=1.0)) == 0
        @test length(multisine(0, frequencies=[1.0], Ts=0.01)) == 0
        @test length(step_signal(0, step_time=0.5, Ts=0.1)) == 0
    end
    
    @testset "Signal Properties for System ID" begin
        # Test that generated signals have properties useful for system identification
        
        N = 1000
        Ts = 0.01
        
        # PRBS should have good autocorrelation properties
        u_prbs = prbs(N)
        autocorr = xcorr(u_prbs, u_prbs)
        # Peak should be at zero lag
        max_idx = argmax(autocorr)
        @test abs(max_idx - length(autocorr)÷2 - 1) <= 1
        
        # Chirp should cover the specified frequency range
        u_chirp = chirp(N, Ts=Ts, f0=0.1, f1=10.0)
        U_chirp = abs.(fft(u_chirp))
        freq_axis = (0:N-1) / (N*Ts)
        
        # Test that chirp signal has broad spectral content (not just DC)
        @test sum(U_chirp[2:end]) > 0.8 * sum(U_chirp)  # Most energy not in DC
        
        # Step signal should have a clear step response
        step_time_sec = (N÷2) * Ts  # Convert sample to time
        u_step = step_signal(N, step_time=step_time_sec, Ts=Ts)
        step_sample = round(Int, step_time_sec / Ts) + 1
        @test all(u_step[1:step_sample-1] .== 0)
        @test all(u_step[step_sample:end] .== 1)
        
        # Multisine should have controlled spectral content
        freqs = [0.5, 2.0, 5.0]
        u_multi = multisine(N, frequencies=freqs, Ts=Ts)
        U_multi = abs.(fft(u_multi))
        
        # Energy should be concentrated at the specified frequencies
        for target_freq in freqs
            _, idx = findmin(abs.(freq_axis .- target_freq))
            nearby_energy = sum(U_multi[max(1,idx-2):min(length(U_multi),idx+2)])
            @test nearby_energy > 0.1 * sum(U_multi)
        end
    end
end