%% FYP PD DATA PIPELINE: SCENE VISUALIZER
% Reads and visualizes a specific scene from the generated HDF5 shard.
% Features per-channel bounding boxes and synchronized TDOA zooming.

clear; clc; close all;

%% 1. Configuration
% Which scene do you want to view? (0-indexed based on our Python schema)
target_scene_id = 0; 

% Define file path (resolves to data/01_raw/ based on your folder structure)
data_dir = fullfile('..','..', 'data', '01_raw');
filename = fullfile(data_dir, 'synth_shard_01.h5');

%% 2. Check File and Read Metadata
if ~isfile(filename)
    error('File not found: %s\nPlease run the Scene Generator first.', filename);
end

fprintf('Loading metadata from %s...\n', filename);
% Read our self-documenting attributes!
dt = h5readatt(filename, '/', 'time_resolution_s');
num_sensors = h5readatt(filename, '/', 'num_sensors');
scene_duration = h5readatt(filename, '/', 'scene_duration_s');
total_scenes = h5readatt(filename, '/', 'num_scenes');

if target_scene_id >= total_scenes || target_scene_id < 0
    error('Invalid scene ID. File contains scenes 0 to %d.', total_scenes - 1);
end

%% 3. Load Data for the Specific Scene
fprintf('Loading Scene %d...\n', target_scene_id);

% Calculate Time Vector
N_scene = round(scene_duration / dt) + 1;
t_scene = (0:N_scene-1) * dt;

% SMART LOADING: We use 'start' and 'count' to only load the specific scene into RAM.
% MATLAB uses 1-based indexing for h5read arrays.
start_idx = [1, 1, target_scene_id + 1];
count_idx = [N_scene, num_sensors, 1];
scene_data = h5read(filename, '/scenes', start_idx, count_idx); 
% scene_data shape is now [N_scene, num_sensors]

% Read all labels
all_labels = h5read(filename, '/labels'); 

% Filter labels for this specific scene
% Format: [Scene_ID, Channel_ID, Class_ID, Start_Idx, End_Idx]
scene_labels = all_labels(all_labels(:, 1) == target_scene_id, :);

%% 4. Plotting
figure('Name', sprintf('FYP Visualizer - Scene %d', target_scene_id), ...
       'Position', [100, 100, 1200, 800]);

colors = lines(num_sensors); % Standard color palette

for ch = 1:num_sensors
    subplot(num_sensors, 1, ch);
    
    % Plot the continuous time-series for this sensor
    plot(t_scene * 1e6, scene_data(:, ch), 'Color', colors(ch, :), 'LineWidth', 1);
    hold on;
    
    % Find labels specifically for this channel (Channel_ID is 0-indexed!)
    ch_labels = scene_labels(scene_labels(:, 2) == (ch - 1), :);
    
    % Draw Bounding Boxes
    y_limits = ylim;
    for i = 1:size(ch_labels, 1)
        class_id = ch_labels(i, 3);
        idx_start = ch_labels(i, 4);
        idx_end = ch_labels(i, 5);
        
        % Convert indices to time in microseconds
        t_start = t_scene(idx_start) * 1e6;
        t_end = t_scene(idx_end) * 1e6;
        
        % Define box color and label based on class
        if class_id == 0
            box_color = 'r'; % PD1 = Red
            class_name = 'PD1';
        else
            box_color = 'b'; % PD2 = Blue
            class_name = 'PD2';
        end
        
        % Draw the rectangle bounding box
        rectangle('Position', [t_start, y_limits(1), t_end - t_start, y_limits(2) - y_limits(1)], ...
                  'EdgeColor', box_color, 'LineWidth', 1.5, 'LineStyle', '--');
                  
        % Add text label slightly above the bottom line
        text(t_start, y_limits(1) + 0.1*(y_limits(2)-y_limits(1)), class_name, ...
             'Color', box_color, 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
    end
    
    % Formatting
    title(sprintf('Sensor %d', ch));
    ylabel('Voltage (V)');
    grid on;
    
    if ch == num_sensors
        xlabel('Time (\mu s)');
    end
end

% CRITICAL FEATURE: Link x-axes so panning/zooming one subplot affects all of them
linkaxes(findall(gcf, 'type', 'axes'), 'x');

sgtitle(sprintf('HDF5 Data Verification: Scene %d', target_scene_id), 'FontSize', 14, 'FontWeight', 'bold');
fprintf('Visualization complete. Use the magnifying glass to zoom in on pulses.\n');