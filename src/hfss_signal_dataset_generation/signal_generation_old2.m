%% FYP PD DATA PIPELINE: MASTER GENERATION CONTROLLER
% Orchestrates the mass generation of HDF5 shards.
% Calculates a global absolute noise floor based on PD occurrence probabilities.

clear; clc;

%% 1. Master Configuration
num_shards = 10;
scenes_per_shard = 10;
output_dir = fullfile('..', "..", 'data', '01_raw');

fprintf('Initializing Master Data Generation...\n');

%% 2. Probabilistic Global Noise Floor Calculation
% Instead of an arbitrary value, we calculate the expected value of the peak
% based on the mathematical probabilities of your PD amplitudes.

% Step A: Define Probabilities
% Based on your generation logic:
% PD1: Amplitude [0.9, 1.1] -> Mean = 1.0 A
% PD2: Amplitude [1.4, 1.6] -> Mean = 1.5 A
% Assuming a 50/50 chance of PD1 vs PD2 occurring:
expected_amplitude_A = (1.0 + 1.5) / 2; % 1.25 A expected average
expected_width_ns = (0.95 + 1.15) / 2;  % 1.05 ns expected average

fprintf('Calculating absolute noise floor based on expected %.2f A pulse...\n', expected_amplitude_A);

% Step B: Dry-Run the Physics to get Voltage
try
    % Load a baseline transfer function (e.g., PD1 to Sensor 1)
    S_data = sparameters('FYP_Sim_Actual_Separated_Ports_PD1_S1.s2p');
    Z_data = s2z(S_data.Parameters, S_data.Impedance);
    Z_transfer = squeeze(Z_data(2, 1, :)) ./ (1 + (squeeze(Z_data(2, 2, :)) / 50));
    
    % Quick FFT Setup
    dt = 0.01e-9;
    t_base = 0:dt:1000e-9;
    N_fft = length(t_base);
    Fs = 1 / dt;
    
    % Generate the Expected Average Pulse
    t0 = 70e-9;
    sigma = (expected_width_ns * 1e-9) / 2.355; 
    i_avg = expected_amplitude_A * exp(-((t_base - t0).^2) / (2 * sigma^2));
    I_freq = fft(i_avg, N_fft);
    
    % Interpolate Z-parameters
    freq_Hz = S_data.Frequencies;
    f_half = (0:floor(N_fft/2)) * (Fs/N_fft);
    Z_half = interp1(freq_Hz, Z_transfer, f_half, 'linear', 0).'; 
    Z_tf_fft = zeros(1, N_fft);
    Z_tf_fft(1:length(Z_half)) = Z_half;
    Z_tf_fft(length(Z_half)+1:end) = conj(flip(Z_half(2:ceil(N_fft/2))));
    
    V_out_freq = I_freq .* Z_tf_fft;
    
    % High-Pass Filter and IFFT
    cutoff_bins = round(10e6 / (Fs/N_fft)); 
    V_out_freq(1:cutoff_bins) = 0;
    V_out_freq(end-cutoff_bins+1:end) = 0; 
    
    % The Holy Grail: Our Global Reference Peak Voltage
    ref_peak_v = max(abs(real(ifft(V_out_freq))));
    fprintf('Success. Global Reference Peak Voltage locked at: %.5f V\n\n', ref_peak_v);
    
catch ME
    % Fallback safety net in case the S-parameter file isn't in the root dir
    ref_peak_v = 0.005; 
    fprintf('S-parameter file missing for physics projection. Using default reference: %.5f V\n\n', ref_peak_v);
end

%% 3. Execute the Data Factory
fprintf('Orchestrating generation of %d shards (%d scenes each)...\n', num_shards, scenes_per_shard);
fprintf('Total expected duration generated: %.1f milliseconds of continuous data.\n\n', ...
        (num_shards * scenes_per_shard * 5000e-9) * 1000);

% Loop through and call your function
for i = 1:num_shards
    generate_pd_shard(i, scenes_per_shard, output_dir, ref_peak_v);
end

fprintf('=== Master Generation Complete! %d total scenes saved to %s ===\n', num_shards * scenes_per_shard, output_dir);