%% FYP PD DATA PIPELINE: SCENE VISUALIZER (V4 - 7-Column Label Support)
% Reads and visualizes a specific scene from the generated HDF5 shard.
% Now compatible with global_pulse_id and dynamic thresholding labels.

clear; clc; close all;

%% 1. Configuration
% Which scene do you want to view? (0-indexed based on Python schema)
target_scene_id = 4; 

% Update this directory to the latest generated folder in your /data/raw/synthesised/
data_dir = fullfile('..','..', 'data', 'raw', 'synthesised', '20260427_170034_sy-ShmH-ShmH');
filename = fullfile(data_dir, 'synth_shard_01.h5');

%% 2. Check File and Read Metadata
if ~isfile(filename)
    error('File not found: %s\nPlease check your folder name or run the Master Controller.', filename);
end

fprintf('Loading metadata from %s...\n', filename);
dt = h5readatt(filename, '/', 'time_resolution_s');
num_sensors = h5readatt(filename, '/', 'num_sensors');
scene_duration = h5readatt(filename, '/', 'scene_duration_s');
total_scenes = h5readatt(filename, '/', 'num_scenes');

if target_scene_id >= total_scenes || target_scene_id < 0
    error('Invalid scene ID. File contains scenes 0 to %d.', total_scenes - 1);
end

%% 3. Load Data for the Specific Scene
fprintf('Loading Scene %d...\n', target_scene_id);
N_scene = round(scene_duration / dt) + 1;
t_scene = (0:N_scene-1) * dt;

% Load specific scene (1-indexed for MATLAB h5read)
start_idx = [1, 1, target_scene_id + 1];
count_idx = [N_scene, num_sensors, 1];
scene_data = h5read(filename, '/scenes', start_idx, count_idx); 

% Read labels
% NEW Format: [Scene_ID, Channel_ID, Class_ID, Pulse_Instance_ID, TOA_Index, Start_Idx, End_Idx]
all_labels = h5read(filename, '/labels'); 

% Filter for specific scene
scene_labels = all_labels(all_labels(:, 1) == target_scene_id, :);

%% 4. Plotting
figure('Name', sprintf('FYP Visualizer - Scene %d', target_scene_id), ...
       'Position', [100, 100, 1200, 850], 'Color', 'w');
colors = lines(num_sensors);

for ch = 1:num_sensors
    ax(ch) = subplot(num_sensors, 1, ch);
    
    % Plot signal
    plot(t_scene * 1e6, scene_data(:, ch), 'Color', colors(ch, :), 'LineWidth', 0.8);
    hold on;
    
    % Filter labels for this channel (0-indexed Column 2)
    ch_labels = scene_labels(scene_labels(:, 2) == (ch - 1), :);
    
    y_limits = ylim;
    for i = 1:size(ch_labels, 1)
        class_id   = ch_labels(i, 3); % Column 3: Class
        pulse_inst = ch_labels(i, 4); % Column 4: Pulse_Instance_ID
        idx_start  = ch_labels(i, 6); % Column 6: Start_Idx
        idx_end    = ch_labels(i, 7); % Column 7: End_Idx
        
        t_start = t_scene(idx_start) * 1e6;
        t_end   = t_scene(idx_end) * 1e6;
        
        box_color = 'r'; if class_id == 1; box_color = 'b'; end
        
        % Draw Bounding Box
        rectangle('Position', [t_start, y_limits(1), t_end - t_start, y_limits(2) - y_limits(1)], ...
                  'EdgeColor', box_color, 'LineWidth', 1.2, 'LineStyle', '--');
                  
        % Label with Class and Global Pulse Instance ID
        text(t_start, y_limits(2), sprintf('  ID:%d (C%d)', pulse_inst, class_id), ...
             'Color', box_color, 'FontSize', 8, 'FontWeight', 'bold', 'VerticalAlignment', 'top');
    end
    
    title(sprintf('Sensor Channel %d', ch-1), 'FontWeight', 'normal');
    ylabel('V'); grid on;
end

xlabel('Time (\mu s)');
linkaxes(ax, 'x'); % Sync all sensors for TDOA analysis
sgtitle(sprintf('Data Verification: Scene %d (RootID: %s)', target_scene_id, ...
    h5readatt(filename, '/', 'root_id')), 'FontSize', 12);

fprintf('Visualization complete. Zooming into any channel will sync all sensors.\n');