%% Digital Twin (simulated) data generation
clear
clc

%% User defined area and grids * Total grid = [(end-start)/step_size + 1] * Y * Z
% Define rectangle coordinates ranges
X_range = [1, 11];
Y_range = [1, 11];
Z_range = [0.1, 2]; % Assuming Z coordinate ranges from 0.1 to 2 meters

% Define the grid or step size for traversing through the area within the rectangle
step_size_X = 0.4;
step_size_Y = 0.4;
step_size_Z = 0.4; % Adjust step size for Z coordinate (0.5 meters in this case)

% Sensor coordinates
sensor_number = {[1, 3, 1.4], [1, 5, 1.8], [1, 7, 1.4], [1, 9, 1.8]};

% Define speed of EM waves
speed_of_EM = 299792458; % Speed of electromagnetic waves in meters per second

% Compute the dimensions of the grid
grid_x = ceil((max(X_range) - min(X_range)) / step_size_X) + 1;
grid_y = ceil((max(Y_range) - min(Y_range)) / step_size_Y) + 1;
grid_z = ceil((max(Z_range) - min(Z_range)) / step_size_Z) + 1;

% Initialize arrays to store results and coordinates
data = cell(grid_x * grid_y * grid_z, 6);
num_of_sensors = 4;

new_data = cell(grid_x * grid_y * grid_z, 6); 

index = 1;
% Nested loops to iterate through all points within the rectangle area
for x_index = 1:grid_x
    for y_index = 1:grid_y
        for z_index = 1:grid_z

            % Clear every iteration to prevent unintended accumulations of data.
            distance = zeros(1, num_of_sensors);
            Theory_Time = zeros(1, num_of_sensors);
            Theory_TDOA_S1 = zeros(1, num_of_sensors);
            Theory_TDOA_S2 = zeros(1, 4);
            Theory_TDOA_S3 = zeros(1, 4);
            Theory_TDOA_S4 = zeros(1, 4);

            % Control PD location
            PD_pointX = min(X_range) + (x_index - 1) * step_size_X; % x-coordinate of the current point
            PD_pointY = min(Y_range) + (y_index - 1) * step_size_Y; % y-coordinate of the current point
            PD_pointZ = min(Z_range) + (z_index - 1) * step_size_Z; % z-coordinate of the current point

            data{index, 1} = PD_pointX;
            data{index, 2} = PD_pointY;
            data{index, 3} = PD_pointZ;

            new_data{index, 1} = PD_pointX;
            new_data{index, 2} = PD_pointY;
            new_data{index, 3} = PD_pointZ;

            % Calculate distances from the current point to each corner of the rectangle
            for i = 1:num_of_sensors % 4/13
                SensorX = sensor_number{i}(1); % x-coordinate of the corner
                SensorY = sensor_number{i}(2); % y-coordinate of the corner
                SensorZ = sensor_number{i}(3); % z-coordinate of the corner

                % Calculating distance from the current point to the current corner
                distance(i) = sqrt((PD_pointX - SensorX)^2 + (PD_pointY - SensorY)^2 + (PD_pointZ - SensorZ)^2);

                % Calculating theoretical time it would take for EM wave to travel from the current point to the current corner
                Theory_Time(i) = distance(i) / speed_of_EM;       
            end

            for i = 1:num_of_sensors
                % Calculating theoretical time difference of arrival between the first corner and the current corner
                Theory_TDOA_S1(i) = Theory_Time(i) - Theory_Time(1);
            end

            % % V1:
            data{index, 4} = Theory_TDOA_S1(1, 2) * 1e9; % t21 (no t11)
            data{index, 5} = Theory_TDOA_S1(1, 3) * 1e9; % t31
            data{index, 6} = Theory_TDOA_S1(1, 4) * 1e9; % t41
         
            index = index + 1;            
        end
    end
end

%% Data Augmentation
% Input: A: TDOA and B: Coordinate matrices
A = cell2mat(data(:, 4:6));
B = cell2mat(data(:, 1:3));

range = 0.2;  % Data Augmentation Range, more details refer paper I, II, III
step = 0.1;   % Step size for generating the range

% Precompute range and step values for a single row
steps_per_dim = numel((0 - range):step:(0 + range));
num_combinations_per_row = steps_per_dim^3;  % Total combinations per row
num_total_combinations = num_combinations_per_row * size(A, 1);

% Preallocate the result matrix
all_combinations_sets = zeros(num_total_combinations, 6);

% Create a 3D grid of offsets for the first row
[dx, dy, dz] = ndgrid(-range:step:range, -range:step:range, -range:step:range);

% Flatten offsets into 1D
offsets = [dx(:), dy(:), dz(:)];

% Loop through each row of A
row_start = 1;
for setIdx = 1:size(A, 1)
    % Extract the current row of A and B
    current_set = A(setIdx, :);
    additional_row = B(setIdx, :);
    
    % Add offsets to generate combinations for the current row of A
    combinations = offsets + current_set;
    
    % Add the corresponding row of B
    complete_combinations = [combinations, repmat(additional_row, size(combinations, 1), 1)];
    
    % Store results in the preallocated matrix
    row_end = row_start + num_combinations_per_row - 1;
    all_combinations_sets(row_start:row_end, :) = complete_combinations;
    row_start = row_end + 1;
end

%% Prepare Training & Validation Dataset (70%:30%)
data = cell2mat(data);

% If using data augmentation, comment Option 1 and uncomment Option 2:
% Option 1:
location_coordinates = data(:, 1:3);
TDOA_data = data(:, 4:6); 

% Option 2:
% location_coordinates = all_combinations_sets(:, 4:6);
% TDOA_data = all_combinations_sets(:, 1:3);

% Randomly shuffle the data
rng(1); % For reproducibility
idx = randperm(size(TDOA_data, 1));

% Calculate the split indices
splitIdx = round(0.70 * length(idx));

% Split the data into training and validation sets
train_TDOA_data = TDOA_data(idx(1 : splitIdx), :);
train_location_coordinates = location_coordinates(idx(1 : splitIdx), :);
val_TDOA_data = TDOA_data(idx(splitIdx + 1 : end), :);
val_location_coordinates = location_coordinates(idx(splitIdx + 1 : end), :);

%% Save to .mat

% Specify the file names for the .mat files at here
matFileName_Train_Coordinates = '1. 33_Train_Coordinates.mat';
matFileName_Train_TDOA = '2. 33_Train_TDOA.mat';
matFileName_Val_Coordinates = '3. 33_Val_Coordinates.mat';
matFileName_Val_TDOA = '4. 33_Val_TDOA.mat';

% Save the variables to .mat files
save(matFileName_Train_Coordinates, 'train_location_coordinates');
save(matFileName_Train_TDOA, 'train_TDOA_data');
save(matFileName_Val_Coordinates, 'val_location_coordinates');
save(matFileName_Val_TDOA, 'val_TDOA_data');

disp('Data saved to .mat file successfully.');