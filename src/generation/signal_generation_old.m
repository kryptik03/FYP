%% FYP PD DATA PIPELINE: SCENE GENERATOR (V2)
% Features: 3D Coordinate TDOA, Dynamic Parameters, Multi-Scene Batching
clear; clc;
%% 1. Configuration & Parameters
num_scenes = 10;               % Number of 5us scenes to generate in this batch
num_pd_sources = 1;            
num_sensors = 3;               
buffer_ns = 50;                % Buffer for TDOA cross-correlation (ns)
label_buffer_ns = 10;          % Tighter buffer specifically for bounding box labels (ns)

% Propagation Physics
c = 3e8;                       % Speed of light in vacuum (m/s)
v_prop = c;  % Velocity of propagation in air
% 3D Coordinates (x, y, z) in meters
coords_PD = [3.7, 5.0, 0.1;   % PD1
             3.7, 7.0, 0.1];  % PD2
coords_S =  [1.0, 3.0, 1.4;   % S1
             1.0, 5.0, 1.8;   % S2
             1.0, 7.0, 1.4;   % S3
             1.0, 9.0, 1.8];  % S4
% Time Domain Settings
dt = 0.01e-9;                  % 10 picosecond resolution
scene_duration = 5000e-9;      % 5 microseconds total canvas
t_scene = 0:dt:scene_duration;
N_scene = length(t_scene);

% FIX: Extended base duration to 3us to prevent FFT circular wraparound
% changed back to 1us
t_base = 0:dt:1000e-9;         
N_fft = length(t_base);
Fs = 1 / dt;
f_fft = (0:N_fft-1) * (Fs/N_fft); 
N_half = floor(N_fft/2) + 1;
f_half = f_fft(1:N_half);
% Pre-allocate the Master Array for Python Compatibility [N_scene, Sensors, Scenes]
% h5py will read this as (num_scenes, num_sensors, N_scene)
batch_scenes = zeros(N_scene, num_sensors, num_scenes);
batch_labels = []; % Format: [Scene_ID, Channel_ID, Class_ID, Start_Idx, End_Idx]
%% 2. Pre-Load S-Parameters & Calculate Transfer Impedances
fprintf('Pre-loading S-parameter files...\n');
Z_transfer_lib = cell(num_pd_sources, num_sensors); % cells are arrays of different data types
for pd = 1:num_pd_sources
    for s = 1:num_sensors
        filename = sprintf('FYP_Sim_Actual_Separated_Ports_PD%d_S%d.s2p', pd, s);
        if isfile(filename)
            S_data = sparameters(filename);
            freq_Hz = S_data.Frequencies; 
            Z_data = s2z(S_data.Parameters, S_data.Impedance);
            Z_transfer = squeeze(Z_data(2, 1, :)) ./ (1 + (squeeze(Z_data(2, 2, :)) / 50));
            % Squeeze turns the 1x1xn into n column vector (matlab is
            % column-major. 
            
            % Interpolate for IFFT
            Z_half = interp1(freq_Hz, Z_transfer, f_half, 'linear', 0).'; 
            Z_tf_fft = zeros(1, N_fft);
            Z_tf_fft(1:N_half) = Z_half;
            Z_tf_fft(N_half+1:end) = conj(flip(Z_half(2:ceil(N_fft/2))));
            Z_transfer_lib{pd, s} = Z_tf_fft;
        else
            warning('File %s not found. Proceeding with caution.', filename);
        end
    end
end
%% 3. Generate Scenes
fprintf('Generating %d Scenes...\n', num_scenes);
buffer_idx = round((buffer_ns * 1e-9) / dt);
label_buffer_idx = round((label_buffer_ns * 1e-9) / dt);

for scene_idx = 1:num_scenes
    % Initialize an empty 4-channel canvas for this scene
    current_scene = zeros(num_sensors, N_scene);
    
    % Randomize number of pulses (0 to 3)
    num_pulses = randi([0, 3]);
    
    for p = 1:num_pulses
        % Pick PD source
        active_pd = randi([1, num_pd_sources]);
        
        % Dynamic Parameters based on PD Type
        if active_pd == 1
            amp = 0.9 + (0.2) * rand();       % [0.9, 1.1] A
            width_ns = 0.9 + (0.1) * rand();  % [0.9, 1.0] ns
        else
            amp = 1.4 + (0.2) * rand();       % [1.4, 1.6] A
            width_ns = 1.1 + (0.1) * rand();  % [1.1, 1.2] ns
        end
        
        % Generate the Unique Base Pulse & FFT
        % FIX: Shifted pulse center deep into the window (50ns) to absorb any acausal HFSS phase shifts
        t0 = 50e-9;   
        sigma = (width_ns * 1e-9) / 2.355; 
        i_t = amp * exp(-((t_base - t0).^2) / (2 * sigma^2));
        I_freq = fft(i_t, N_fft);
        
        % Calculate Physics-Based TDOA
        % Distance from active PD to all 4 sensors
        distances = sqrt(sum((coords_S - coords_PD(active_pd, :)).^2, 2)); % returns a column vector of the distances to each sensor
        tof = distances / v_prop;                  % Time of Flight (seconds)
        tdoa_sec = tof - min(tof);                 % Normalize to first arriving signal
        tdoa_idx = round(tdoa_sec / dt);           % Convert to array indices
        
        % Determine safe injection point
        max_tdoa_idx = max(tdoa_idx);
        start_idx = randi([buffer_idx + 1, N_scene - N_fft - max_tdoa_idx - buffer_idx]);
        
        % Inject into all 4 channels
        class_label = active_pd - 1; 
        for ch = 1:num_sensors
            Z_tf = Z_transfer_lib{active_pd, ch};
            V_out_freq = I_freq .* Z_tf;
            
            % High-Pass Filter
            cutoff_bins = round(10e6 / (Fs/N_fft)); 
            V_out_freq(1:cutoff_bins) = 0;
            V_out_freq(end-cutoff_bins+1:end) = 0; 
            
            v_out = real(ifft(V_out_freq));
            
            % Superimpose with exact TDOA shift
            idx_in = start_idx + tdoa_idx(ch);
            idx_end = idx_in + N_fft - 1;
            current_scene(ch, idx_in:idx_end) = current_scene(ch, idx_in:idx_end) + v_out;
            
            % Autonomous Labeling (Per-Channel with 10ns buffer)
            start_ch = idx_in - label_buffer_idx;
            end_ch = idx_end + label_buffer_idx;
            
            % Format: [Scene_ID, Channel_ID, Class_ID, Start_Idx, End_Idx]
            % Note: We use scene_idx - 1 and ch - 1 so Python gets 0-indexed IDs!
            batch_labels = [batch_labels; (scene_idx - 1), (ch - 1), class_label, start_ch, end_ch];
        end
    end
    
    % Add Environmental Noise
    current_scene = awgn(current_scene, 20, 'measured'); % Gaussian
    fm_noise = 0.05 * max(max(abs(current_scene))) * sin(2 * pi * 100e6 * t_scene);
    current_scene = current_scene + fm_noise;
    
    % Store in Master Array (Transposed for Python: [Time, Sensor, Scene])
    batch_scenes(:, :, scene_idx) = current_scene.'; % I checked and verified, seems ok
end
%% 4. Export to HDF5 Batch
% Set up the relative path to data/01_raw/
output_dir = fullfile('..','..', 'data', '01_raw');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_file = fullfile(output_dir, 'synth_shard_01.h5');

if isfile(output_file); delete(output_file); end
fprintf('Writing %d scenes to %s...\n', num_scenes, output_file);

% Create and write main datasets
h5create(output_file, '/scenes', size(batch_scenes), 'Datatype', 'double');
h5write(output_file, '/scenes', batch_scenes);
if ~isempty(batch_labels)
    h5create(output_file, '/labels', size(batch_labels), 'Datatype', 'double');
    h5write(output_file, '/labels', batch_labels);
end

%% 5. Append Global Metadata & Attributes
fprintf('Attaching self-documenting metadata...\n');

% Global File Attributes
h5writeatt(output_file, '/', 'creation_date', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
h5writeatt(output_file, '/', 'sampling_frequency_Hz', Fs);
h5writeatt(output_file, '/', 'time_resolution_s', dt);
h5writeatt(output_file, '/', 'scene_duration_s', scene_duration);
h5writeatt(output_file, '/', 'num_scenes', num_scenes);
h5writeatt(output_file, '/', 'num_sensors', num_sensors);
h5writeatt(output_file, '/', 'max_pulses_per_scene', 3);
h5writeatt(output_file, '/', 'v_prop_m_s', v_prop);
h5writeatt(output_file, '/', 'fft_resolution_Hz', Fs/N_fft);
h5writeatt(output_file, '/', 'tdoa_buffer_ns', buffer_ns);
h5writeatt(output_file, '/', 'label_buffer_ns', label_buffer_ns);

% Specific Dataset Attributes (Helps massively when importing to Python)
h5writeatt(output_file, '/scenes', 'matlab_shape', sprintf('[%d, %d, %d]', size(batch_scenes)));
h5writeatt(output_file, '/scenes', 'python_h5py_shape', sprintf('(%d, %d, %d)', num_scenes, num_sensors, N_scene));
h5writeatt(output_file, '/scenes', 'dimension_1', 'Python Dim 0: Scene ID');
h5writeatt(output_file, '/scenes', 'dimension_2', 'Python Dim 1: Sensor Channel');
h5writeatt(output_file, '/scenes', 'dimension_3', 'Python Dim 2: Time (dt)');

h5writeatt(output_file, '/labels', 'column_1', 'Scene_ID (0-indexed)');
h5writeatt(output_file, '/labels', 'column_2', 'Channel_ID (0-indexed)');
h5writeatt(output_file, '/labels', 'column_3', 'Class_ID (0=PD1, 1=PD2)');
h5writeatt(output_file, '/labels', 'column_4', 'Start_Idx');
h5writeatt(output_file, '/labels', 'column_5', 'End_Idx');

fprintf('Batch generation complete.\n');