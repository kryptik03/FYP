%% Load the model from the saved file
clear
clc

loaded_data = load('best_model_33.mat'); % DNN: best_model_33.mat; Augmented DNN: best_model_36_0.1gs_0.2r.mat
net = loaded_data.net; % Access the model object from the struct % <====== .net or .best_model or .new_net
info = loaded_data.info; % Access the model object from the struct
disp('Model load successfully.');

%% 1. Call Sensors_Data - Contained: the 4 sensor coordinates, Speed of EM wave, PD TDOA & PD coordinates for benchmarking.
[receiver1, receiver2, receiver3, receiver4, speed_of_EM, data_predict] = func_Sensors_Data();

% Loop through all the TDOA in data_predict
num_of_loop = size(data_predict, 1);

FOR_EXCEL = cell(num_of_loop, 5); % <=======================

for data_set = 1:num_of_loop
    StartTime = datetime('now');

    % Define the PD coordinates
    pointX = data_predict(data_set, 1);    pointY = data_predict(data_set, 2);    pointZ = data_predict(data_set, 3);
    true_source_location = [pointX, pointY, pointZ];

    % TDOA_N = [t21, t31, t41];
    TDOA_N = [data_predict(data_set, 4), data_predict(data_set, 5), data_predict(data_set, 6)];

    % Prediction START HERE:         
    estimated_location = predict(net, TDOA_N); % <=============

    %% 6. Compute_Error
    [Euclidean_Distance, percentage_error] = func_Compute_Error(true_source_location, estimated_location, receiver1, receiver2, receiver3, receiver4);
           
    %% 8. Display_results
    func_Display_Results(estimated_location, true_source_location, Euclidean_Distance, percentage_error);

    EndTime = datetime('now');
    durationBetween = EndTime - StartTime;

    true_source_location = arrayfun(@(x) sprintf('%.2f', x), true_source_location, 'UniformOutput', false);

    % Format Numbers as Strings Before Export
    % With round off
    % Round to 4 significant figures
    FOR_EXCEL{data_set, 1} = round(percentage_error, 2);
    FOR_EXCEL{data_set, 2} = round(Euclidean_Distance, 3);
    FOR_EXCEL{data_set, 3} = round(seconds(durationBetween), 2);
    FOR_EXCEL{data_set, 4} = estimated_location;
    FOR_EXCEL{data_set, 5} = strjoin(true_source_location, ', ');
end

% Save the data
% save('FOR_EXCEL_DNN.mat', 'FOR_EXCEL');
