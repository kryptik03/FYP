%% CWRU Bearing Dataset Visualization
% This script loads a CWRU .mat file, automatically identifies the 
% vibration time-series variables, and plots them with proper time axes.

clear; clc; close all;

% =========================================================================
% 1. Configuration
% =========================================================================
% Update this path to where your downloaded CWRU .mat file is located
filename = fullfile('..', '..', 'data', 'unprocessed_bearing', '12k_Fan_End', 'IR021_2.mat');

% CWRU data is sampled at either 12 kHz or 48 kHz. 
% (12,000 is the standard for most drive-end bearing experiments).
Fs = 12000; 

% =========================================================================
% 2. Load and Parse Data
% =========================================================================
if ~isfile(filename)
    error('File "%s" not found. Please verify the path.', filename);
end

fprintf('Loading %s...\n', filename);
cwru_data = load(filename);
fields = fieldnames(cwru_data);

% =========================================================================
% 3. Extract and Plot
% =========================================================================
figure('Name', sprintf('CWRU Bearing Data: %s', filename), ...
       'Position', [100, 100, 1000, 600]);

plot_count = 0;
% The actual vibration arrays always end in '_time'
time_vars = fields(contains(fields, '_time'));

if isempty(time_vars)
    error('No vibration data found. CWRU variables should contain "_time".');
end

num_plots = length(time_vars);

for i = 1:num_plots
    var_name = time_vars{i};
    signal = cwru_data.(var_name);
    
    % Create a time vector for the x-axis based on the sampling frequency
    t = (0:length(signal)-1) / Fs;
    
    % Create a dynamic subplot based on how many sensors were recorded
    subplot(num_plots, 1, i);
    plot(t, signal, 'Color', [0 0.4470 0.7410]); % Standard MATLAB blue
    
    % Format the title. Underscores make MATLAB subscript things, 
    % so we replace them with spaces for readability.
    clean_title = strrep(var_name, '_', ' ');
    title(clean_title, 'FontWeight', 'bold');
    
    xlabel('Time (Seconds)');
    ylabel('Acceleration (g)');
    grid on;
    axis tight;
    
    % Optional: Zoom in on the first 0.1 seconds to see the actual waveform
    % xlim([0, 0.1]); 
end

fprintf('Successfully plotted %d sensor tracks.\n', num_plots);