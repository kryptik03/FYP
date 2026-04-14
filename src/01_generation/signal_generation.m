%% FYP PD DATA PIPELINE: SCENE GENERATOR (PHASE 1 - UPDATED)
% Simulates a 4-sensor canvas using unique S-parameters for each sensor 
% and each PD source (PD1, PD2) to model true propagation distortion.

clear; clc;

%% 1. Configuration & Parameters
num_pd_sources = 2;            % PD1 and PD2
num_sensors = 4;               % S1, S2, S3, S4
num_pulses_in_scene = 3;       % Number of pulses to inject (creates overlaps)
buffer_ns = 50;                % Buffer for TDOA cross-correlation (ns)

% Time Domain Settings
dt = 0.01e-9;                  % 10 picosecond resolution
scene_duration = 5000e-9;      % 5 microseconds total canvas
t_scene = 0:dt:scene_duration;
N_scene = length(t_scene);

% Base Pulse Settings (Gaussian)
t_base = 0:dt:1000e-9;         
AMPLITUDE = 1.0;               
TIME_WIDTH_NS = 1;           
t0 = TIME_WIDTH_NS * 1.5e-9;   
sigma = (TIME_WIDTH_NS * 1e-9) / 2.355; 
i_t = AMPLITUDE * exp(-((t_base - t0).^2) / (2 * sigma^2));

% FFT Setup for the Base Pulse
N_fft = length(t_base);          
Fs = 1 / dt;                      
f_fft = (0:N_fft-1) * (Fs/N_fft); 
I_freq = fft(i_t, N_fft); 
N_half = floor(N_fft/2) + 1;
f_half = f_fft(1:N_half);

%% 2. Pre-Load S-Parameters & Calculate Transfer Impedances
fprintf('Pre-loading %d S-parameter files...\n', num_pd_sources * num_sensors);

% Cell array to store the 8 unique FFT Transfer Functions
% Z_transfer_lib{pd_id, sensor_id}
Z_transfer_lib = cell(num_pd_sources, num_sensors);

for pd = 1:num_pd_sources
    for s = 1:num_sensors
        % Construct the dynamic filename
        filename = sprintf('FYP_Sim_Actual_Separated_Ports_PD%d_S%d.s2p', pd, s);
        
        % Load and convert to Z-parameters
        S_data = sparameters(filename);
        freq_Hz = S_data.Frequencies(1:91); % Assuming 91 points based on your original code
        Z_data = s2z(S_data.Parameters(:,:, 1:91), S_data.Impedance);
        
        Z21 = squeeze(Z_data(2, 1, :)); 
        Z22 = squeeze(Z_data(2, 2, :));
        
        % Calculate Exact Transfer Impedance
        R_load = 50; 
        Z_transfer = Z21 ./ (1 + (Z22 / R_load));
        
        % Interpolate for IFFT
        Z_half = interp1(freq_Hz, Z_transfer, f_half, 'linear', 0).'; 
        Z_transfer_fft = zeros(1, N_fft);
        Z_transfer_fft(1:N_half) = Z_half;
        Z_transfer_fft(N_half+1:end) = conj(flip(Z_half(2:ceil(N_fft/2))));
        
        % Store in the library
        Z_transfer_lib{pd, s} = Z_transfer_fft;
    end
end

%% 3. Construct Canvas & Inject Pulses
sensor_data = zeros(num_sensors, N_scene);
bounding_boxes = []; % Format: [Class_ID, Start_Idx, End_Idx]
buffer_idx = round((buffer_ns * 1e-9) / dt);

fprintf('Injecting %d pulses into the Scene...\n', num_pulses_in_scene);

for i = 1:num_pulses_in_scene
    % Randomly select which PD source fired (1 or 2)
    active_pd = randi([1, num_pd_sources]);
    
    % Randomly pick an injection time
    max_start_idx = N_scene - N_fft - 5000; 
    start_idx = randi([buffer_idx + 1, max_start_idx]);
    
    % Simulate TDOA Delays
    tdoa_delays_ns = [0, rand()*10, rand()*20, rand()*30]; 
    tdoa_idx = round((tdoa_delays_ns * 1e-9) / dt);
    
    % Inject the unique waveform into each sensor channel
    for ch = 1:num_sensors
        % 1. Get the specific transfer function for this PD -> Sensor path
        Z_tf = Z_transfer_lib{active_pd, ch};
        
        % 2. Calculate the unique V_out using Ohm's Law in Freq Domain
        V_out_freq = I_freq .* Z_tf;
        
        % High-Pass Filter (DC Block)
        cutoff_bins = round(10e6 / (Fs/N_fft)); 
        V_out_freq(1:cutoff_bins) = 0;
        V_out_freq(end-cutoff_bins+1:end) = 0; 
        
        % Back to Time Domain
        v_out = real(ifft(V_out_freq));
        
        % 3. Superimpose onto the Canvas with TDOA
        idx_in = start_idx + tdoa_idx(ch);
        idx_end = idx_in + length(v_out) - 1;
        sensor_data(ch, idx_in:idx_end) = sensor_data(ch, idx_in:idx_end) + v_out;
    end
    
    % Autonomous Labeling (Subtract 1 from active_pd so classes are 0 and 1)
    class_label = active_pd - 1; 
    global_start = start_idx - buffer_idx;
    global_end = start_idx + max(tdoa_idx) + length(v_out) - 1 + buffer_idx;
    
    bounding_boxes = [bounding_boxes; class_label, global_start, global_end];
end

%% 4. Augment with Realistic Industrial Noise
fprintf('Adding environmental noise...\n');

% 4a. Gaussian White Noise (Baseline Floor)
SNR_dB = 20; % Signal to Noise Ratio
sensor_data = awgn(sensor_data, SNR_dB, 'measured');

% 4b. Structured Noise (FM/GSM Interference)
fm_freq = 100e6; % 100 MHz FM radio
fm_noise = 0.05 * max(max(abs(sensor_data))) * sin(2 * pi * fm_freq * t_scene);
sensor_data = sensor_data + fm_noise;

%% 5. Export to HDF5 Shard
% Note: In a full generation run, you would loop this script to append scenes.
output_file = 'scene_shard_01.h5';

% Delete file if it exists (for testing)
if isfile(output_file)
    delete(output_file);
end

% Create HDF5 datasets with chunking (Crucial for memory limits in DL training)
h5create(output_file, '/scenes', size(sensor_data), 'ChunkSize', [4, min(10000, N_scene)]);
h5create(output_file, '/labels', size(bounding_boxes));

% Write the data
h5write(output_file, '/scenes', sensor_data);
h5write(output_file, '/labels', bounding_boxes);

fprintf('Scene successfully exported to %s\n', output_file);

%% 6. Plotting to Verify Overlaps and Labels
figure('Name', 'Generated Scene Validation', 'Position', [100, 100, 1000, 600]);
plot(t_scene * 1e6, sensor_data(1, :), 'b');
hold on;

% Draw the bounding boxes to verify labels
for i = 1:size(bounding_boxes, 1)
    x_start = t_scene(bounding_boxes(i, 2)) * 1e6;
    x_end = t_scene(bounding_boxes(i, 3)) * 1e6;
    y_limits = ylim;
    rectangle('Position', [x_start, y_limits(1), x_end - x_start, y_limits(2) - y_limits(1)], ...
              'EdgeColor', 'r', 'LineWidth', 1.5, 'LineStyle', '--');
end

title('Sensor 1: Continuous Stream with Injected Overlaps, Noise, and Labels');
xlabel('Time (\mu s)');
ylabel('Voltage (V)');
grid on;