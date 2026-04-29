%% FYP PD DATA PIPELINE: MASTER GENERATION CONTROLLER
% Orchestrates the mass generation of HDF5 shards.
% Compliant with MLOps DAG Architecture v3.0

clear; clc;
rng('shuffle'); % Seeds the random number generator based on the current time

%% 1. Master Configuration & Lineage Setup
num_shards = 20;
scenes_per_shard = 20;
nickname = 'First try at an actual dataset'; % User-defined nickname

fprintf('Initializing Master Data Generation...\n');

% Generate 4-character RootID
chars = ['a':'z', 'A':'Z', '0':'9'];
root_id = chars(randi(length(chars), 1, 4));

% Generate Timestamp and Folder Path
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
folder_name = sprintf('%s_sy-%s-%s', timestamp, root_id, root_id);
output_dir = fullfile('..', '..', 'data', 'raw', 'synthesised', folder_name);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Create Verbose History Log
history_log = sprintf('Synthetic dataset [Nickname: %s] generated at %s, RootID: %s', ...
    nickname, datestr(now, 'yyyy-mm-dd HH:MM:SS'), root_id);

% Drop a standalone text ledger into the directory
fid = fopen(fullfile(output_dir, 'analysis_history.txt'), 'w');
fprintf(fid, '%s\n', history_log);
fclose(fid);

% =========================================================================
% NEW: SQLite Ledger Integration via System Terminal
% =========================================================================
fprintf('Registering dataset to SQLite Master Ledger...\n');

% Path to your python script
tracker_script = fullfile('..', '..', 'src', 'utils', 'lineage_tracker.py');

% Build the terminal command (using double quotes to handle spaces in strings)
cmd = sprintf('python "%s" --action register_root --origin "sy" --method "generation" --folder_path "%s" --nickname "%s" --history_log "%s" --root_id "%s" --timestamp "%s"', ...
    tracker_script, output_dir, nickname, history_log, root_id, timestamp);

% Execute the command in the OS terminal
[status, cmdout] = system(cmd);

if status == 0
    fprintf('Successfully registered Node %s to SQLite Database.\n', root_id);
else
    warning('SQLite Registration Failed. Python error trace:\n%s', cmdout);
end
% =========================================================================

fprintf('Lineage Established. Target Directory: %s\n', folder_name);

%% 2. Probabilistic Global Noise Floor Calculation
expected_amplitude_A = (1.0 + 1.5) / 2; % 1.25 A expected average
expected_width_ns = (0.95 + 1.15) / 2;  % 1.05 ns expected average

fprintf('Calculating absolute noise floor based on expected %.2f A pulse...\n', expected_amplitude_A);

try
    % Routed to the new touchstone_files directory
    s2p_path = fullfile('..', '..', 'data', 'touchstone_files', 'FYP_Sim_Actual_Separated_Ports_PD1_S1.s2p');
    S_data = sparameters(s2p_path);
    Z_data = s2z(S_data.Parameters, S_data.Impedance);
    Z_transfer = squeeze(Z_data(2, 1, :)) ./ (1 + (squeeze(Z_data(2, 2, :)) / 50));
    
    dt = 0.01e-9;
    t_base = 0:dt:1000e-9;
    N_fft = length(t_base);
    Fs = 1 / dt;
    
    t0 = 70e-9;
    sigma = (expected_width_ns * 1e-9) / 2.355; 
    i_avg = expected_amplitude_A * exp(-((t_base - t0).^2) / (2 * sigma^2));
    I_freq = fft(i_avg, N_fft);
    
    freq_Hz = S_data.Frequencies;
    f_half = (0:floor(N_fft/2)) * (Fs/N_fft);
    Z_half = interp1(freq_Hz, Z_transfer, f_half, 'linear', 0).'; 
    Z_tf_fft = zeros(1, N_fft);
    Z_tf_fft(1:length(Z_half)) = Z_half;
    Z_tf_fft(length(Z_half)+1:end) = conj(flip(Z_half(2:ceil(N_fft/2))));
    
    V_out_freq = I_freq .* Z_tf_fft;
    
    cutoff_bins = round(10e6 / (Fs/N_fft)); 
    V_out_freq(1:cutoff_bins) = 0;
    V_out_freq(end-cutoff_bins+1:end) = 0; 
    
    ref_peak_v = max(abs(real(ifft(V_out_freq))));
    fprintf('Success. Global Reference Peak Voltage locked at: %.5f V\n\n', ref_peak_v);
    
catch ME
    ref_peak_v = 0.005; 
    fprintf('S-parameter file missing for physics projection. Using default reference: %.5f V\n\n', ref_peak_v);
end

%% 3. Execute the Data Factory
fprintf('Orchestrating generation of %d shards (%d scenes each)...\n', num_shards, scenes_per_shard);

% NEW: Initialize the master tracker before the loop starts
current_global_pulse_id = 0; 

for i = 1:num_shards
    % NEW: Pass the ID in, and catch the updated ID when the function finishes
    current_global_pulse_id = generate_pd_shard(i, scenes_per_shard, output_dir, ...
                              ref_peak_v, root_id, nickname, timestamp, history_log, ...
                              current_global_pulse_id);
end

fprintf('=== Master Generation Complete! %d total scenes saved to %s ===\n', num_shards * scenes_per_shard, output_dir);