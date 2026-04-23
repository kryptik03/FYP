% NEW: The function now outputs 'final_pulse_id' and accepts 'starting_pulse_id'
function final_pulse_id = generate_pd_shard(shard_id, num_scenes_per_shard, output_dir, ref_peak_v, root_id, nickname, timestamp, history_log, starting_pulse_id)
    arguments
        shard_id
        num_scenes_per_shard
        output_dir
        ref_peak_v double
        root_id char
        nickname char
        timestamp char
        history_log char
        starting_pulse_id double
    end
    %% FYP PD DATA PIPELINE: SCENE GENERATOR FUNCTION (V4)
    % Features: Dynamic Thresholding, TOA Tracking, Pulse IDs, Verbose Metadata Injection
    
    fprintf('=== Starting Generation for Shard %02d ===\n', shard_id);
    %% 1. Configuration & Parameters
    num_pd_sources = 1;            
    num_sensors = 4;               
    buffer_ns = 50;                % Buffer for safe injection bounds (ns)
    label_buffer_ns = 10;          % Tighter buffer specifically for bounding box labels (ns)
    threshold_pct = 0.05;          % 5% (-26dB) dynamic threshold for bounding box cutoff
    
    % Noise Configuration
    target_snr_db = 20;            % Target SNR relative to a standard 1A reference pulse
    
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
    
    t_base = 0:dt:1000e-9;         
    N_fft = length(t_base);
    Fs = 1 / dt;
    f_fft = (0:N_fft-1) * (Fs/N_fft); 
    N_half = floor(N_fft/2) + 1;
    f_half = f_fft(1:N_half);
    
    % Pre-allocate the Master Array for Python Compatibility
    batch_scenes = zeros(N_scene, num_sensors, num_scenes_per_shard);
    batch_labels = []; % NEW Format: [Scene_ID, Channel_ID, Class_ID, Pulse_Instance_ID, TOA_Index, Start_Idx, End_Idx]
    
    %% 2. Pre-Load S-Parameters & Calculate Transfer Impedances
    Z_transfer_lib = cell(num_pd_sources, num_sensors); 
    for pd = 1:num_pd_sources
        for s = 1:num_sensors
            % Routed to the new touchstone_files directory
            filename = fullfile('..', '..', 'data', 'touchstone_files', sprintf('FYP_Sim_Actual_Separated_Ports_PD%d_S%d_Z_Matrix.s2p', pd, s));
            if isfile(filename)
                Z_data_raw = zparameters(filename);
                freq_Hz = Z_data_raw.Frequencies; 
                Z_data = Z_data_raw.Parameters;
                Z_transfer = squeeze(Z_data(2, 1, :)) ./ (1 + (squeeze(Z_data(2, 2, :)) / 50));
                
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
    
    % Calculate absolute noise standard deviation (RMS) based on voltage ratio
    noise_rms = ref_peak_v * (10^(-target_snr_db / 20));
    fm_amplitude = 0.05 * ref_peak_v;
    
    %% 3. Generate Scenes
    buffer_idx = round((buffer_ns * 1e-9) / dt);
    label_buffer_idx = round((label_buffer_ns * 1e-9) / dt);
    
    % Initialize the counter with the baton passed from the Master Script
    global_pulse_id = starting_pulse_id;
    
    for scene_idx = 1:num_scenes_per_shard
        current_scene = zeros(num_sensors, N_scene);
        num_pulses = randi([0, 3]);
        
        for p = 1:num_pulses
            active_pd = randi([1, num_pd_sources]);
            
            % ⭐ FIX: Increment the physical event ID for every new pulse generated
            global_pulse_id = global_pulse_id + 1; 
            
            % Dynamic Parameters based on PD Type
            if active_pd == 1
                amp = 0.9 + (0.2) * rand();       
                width_ns = 0.9 + (0.1) * rand();  
            else
                amp = 1.4 + (0.2) * rand();       
                width_ns = 1.1 + (0.1) * rand();  
            end
            
            % Shifted pulse center deep into the window (50ns) to absorb acausal shifts
            t0 = 100e-9;   
            sigma = (width_ns * 1e-9) / 2.355; 
            i_t = amp * exp(-((t_base - t0).^2) / (2 * sigma^2));
            I_freq = fft(i_t, N_fft);
            
            % Calculate Physics-Based TDOA
            distances = sqrt(sum((coords_S - coords_PD(active_pd, :)).^2, 2)); 
            tof = distances / v_prop;                  
            tdoa_sec = tof - min(tof);                 
            tdoa_idx = round(tdoa_sec / dt);           
            
            max_tdoa_idx = max(tdoa_idx);
            start_idx = randi([buffer_idx + 1, N_scene - N_fft - max_tdoa_idx - buffer_idx]);
            
            class_label = active_pd - 1; 
            
            for ch = 1:num_sensors
                Z_tf = Z_transfer_lib{active_pd, ch};
                V_out_freq = I_freq .* Z_tf;
                
                % High-Pass Filter
                cutoff_bins = round(10e6 / (Fs/N_fft)); 
                V_out_freq(1:cutoff_bins) = 0;
                V_out_freq(end-cutoff_bins+1:end) = 0; 
                
                v_out = real(ifft(V_out_freq));
                
                % DYNAMIC THRESHOLDING LOGIC
                peak_val = max(abs(v_out));
                cutoff_threshold = threshold_pct * peak_val;
                active_indices = find(abs(v_out) > cutoff_threshold);
                
                if ~isempty(active_indices)
                    local_start = active_indices(1);
                    local_end = active_indices(end);
                else
                    % Fallback in case of numeric failure
                    local_start = 1;
                    local_end = N_fft;
                end
                
                % Superimpose with exact TDOA shift
                idx_in = start_idx + tdoa_idx(ch);
                idx_end = idx_in + N_fft - 1;
                current_scene(ch, idx_in:idx_end) = current_scene(ch, idx_in:idx_end) + v_out;
                
                % ⭐ FIX: Time of Arrival Calculation (Index of the start of the injected physics window)
                toa_idx = idx_in; 
                
                % Autonomous Labeling using the Dynamic Threshold
                global_pulse_start = idx_in + local_start - 1;
                global_pulse_end = idx_in + local_end - 1;
                
                start_ch = global_pulse_start - label_buffer_idx;
                end_ch = global_pulse_end + label_buffer_idx;
                
                % Boundary Safety Check
                start_ch = max(1, start_ch);
                end_ch = min(N_scene, end_ch);
                
                % ⭐ FIX: 7-Column Label Matrix (0-indexed for Python compatibility)
                batch_labels = [batch_labels; (scene_idx - 1), (ch - 1), class_label, (global_pulse_id - 1), toa_idx, start_ch, end_ch];
            end
        end
        
        % CONSTANT ENVIRONMENTAL NOISE INJECTION
        current_scene = current_scene + (noise_rms * randn(num_sensors, N_scene));
        fm_wave = fm_amplitude * sin(2 * pi * 100e6 * t_scene);
        current_scene = current_scene + repmat(fm_wave, num_sensors, 1);
        
        % Store in Master Array 
        batch_scenes(:, :, scene_idx) = current_scene.'; 
    end
    
    %% 4. Export to HDF5 Batch
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Dynamically format the filename (e.g., synth_shard_01.h5)
    filename = sprintf('synth_shard_%02d.h5', shard_id);
    output_file = fullfile(output_dir, filename);
    if isfile(output_file); delete(output_file); end
    
    h5create(output_file, '/scenes', size(batch_scenes), 'Datatype', 'double');
    h5write(output_file, '/scenes', batch_scenes);
    
    if ~isempty(batch_labels)
        h5create(output_file, '/labels', size(batch_labels), 'Datatype', 'double');
        h5write(output_file, '/labels', batch_labels);
    end
    
    %% 5. Append Global Metadata & Attributes
    h5writeatt(output_file, '/', 'creation_date', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    h5writeatt(output_file, '/', 'sampling_frequency_Hz', Fs);
    h5writeatt(output_file, '/', 'time_resolution_s', dt);
    h5writeatt(output_file, '/', 'scene_duration_s', scene_duration);
    h5writeatt(output_file, '/', 'num_scenes', num_scenes_per_shard);
    h5writeatt(output_file, '/', 'num_sensors', num_sensors);
    h5writeatt(output_file, '/', 'shard_id', shard_id);
    h5writeatt(output_file, '/', 'v_prop_m_s', v_prop);
    h5writeatt(output_file, '/', 'threshold_pct', threshold_pct);
    h5writeatt(output_file, '/', 'label_buffer_ns', label_buffer_ns);
    h5writeatt(output_file, '/', 'target_snr_db', target_snr_db);
    h5writeatt(output_file, '/', 'ref_peak_v', ref_peak_v);
    
    % ⭐ FIX: LINEAGE INJECTION (This clears the orange squiggly lines)
    h5writeatt(output_file, '/', 'root_id', root_id);
    h5writeatt(output_file, '/', 'node_id', root_id); % Because this is generation, root_id == node_id
    h5writeatt(output_file, '/', 'nickname', nickname);
    h5writeatt(output_file, '/', 'analysis_history', history_log);
    
    h5writeatt(output_file, '/scenes', 'matlab_shape', sprintf('[%d, %d, %d]', size(batch_scenes)));
    h5writeatt(output_file, '/scenes', 'python_h5py_shape', sprintf('(%d, %d, %d)', num_scenes_per_shard, num_sensors, N_scene));
    h5writeatt(output_file, '/scenes', 'dimension_1', 'Python Dim 0: Scene ID');
    h5writeatt(output_file, '/scenes', 'dimension_2', 'Python Dim 1: Sensor Channel');
    h5writeatt(output_file, '/scenes', 'dimension_3', 'Python Dim 2: Time (dt)');
    
    % ⭐ FIX: Updated Label Definitions
    h5writeatt(output_file, '/labels', 'column_1', 'Scene_ID (0-indexed)');
    h5writeatt(output_file, '/labels', 'column_2', 'Channel_ID (0-indexed)');
    h5writeatt(output_file, '/labels', 'column_3', 'Class_ID (0=PD1, 1=PD2)');
    h5writeatt(output_file, '/labels', 'column_4', 'Pulse_Instance_ID (0-indexed)');
    h5writeatt(output_file, '/labels', 'column_5', 'TOA_Index');
    h5writeatt(output_file, '/labels', 'column_6', 'Start_Idx');
    h5writeatt(output_file, '/labels', 'column_7', 'End_Idx');
    
    % Pass the baton back to the Master Script before the function ends
    final_pulse_id = global_pulse_id;
    
    fprintf('Shard %02d generation complete. Saved to %s\n\n', shard_id, output_file);
end