 clear
clc

%% Load dataset (Training and Validation)
[train_location_coordinates, train_TDOA_data, val_location_coordinates, val_TDOA_data] = func_load_data( ...
    '1. 33_Train_Coordinates.mat', ...
    '2. 33_Train_TDOA.mat', ...
    '3. 33_Val_Coordinates.mat', ...
    '4. 33_Val_TDOA.mat');

disp('Data load successfully.');

%% Define Hyperparameter Grid
num_features = size(train_TDOA_data, 2); % Get the number of features usually 3
num_responses = size(train_location_coordinates, 2); % Get the number of responses usually 2

learn_rates = [0.0001];
layer_sizes = [4096]; % Different sizes for hidden layers
num_hidden_layers = [5]; % Different numbers of hidden layers

% learn_rates = [0.01, 0.001, 0.0001]; % Different initial learning rates
% layer_sizes = [128, 256, 512, 1024, 2048, 4096]; % Different sizes for hidden layers
% num_hidden_layers = [3, 4, 5]; % Different numbers of hidden layers

best_RMSE = Inf; % Initialize best RMSE to infinity
best_model = []; % Initialize best model
best_info = [];

% Initialize arrays to store RMSE values and corresponding hyperparameters
RMSE_values = zeros(numel(layer_sizes), numel(learn_rates), numel(num_hidden_layers));
hyperparameters = zeros(numel(layer_sizes)*numel(learn_rates)*numel(num_hidden_layers), 4); % based on 3 GS parameter + 1 for RMSE = 4

hyperparameters_counter = 1;
total_training_time = zeros(3, 3, 3);

StartTime = datetime('now');
disp(['Start time: ', datestr(StartTime)]); % To show the time of finishing optimization

% Iterate over all combinations of hyperparameters
for learn_idx = 1:numel(learn_rates)
    for layer_idx = 1:numel(layer_sizes)
        for hidden_idx = 1:numel(num_hidden_layers)
            learn_rate = learn_rates(learn_idx);
            layer_size = layer_sizes(layer_idx);
            num_hidden_layer = num_hidden_layers(hidden_idx);

            fprintf('Training model with layer size %d, learning rate %.5f, and %d hidden layers\n', layer_size, learn_rate, num_hidden_layer);

            % Define neural network architecture
            layers = [
                featureInputLayer(num_features) % Input layer with num_features features
                 ];

            % Add additional hidden layers
            for i = 1:num_hidden_layer
                layers = [layers
                    fullyConnectedLayer(layer_size) % Additional hidden layers
                    reluLayer % ReLU activation function: reluLayer, sigmoidLayer, tanhLayer
                    dropoutLayer(0.5)];
            end

            layers = [layers
                fullyConnectedLayer(num_responses) % Output layer with 2 neurons for regression
                 % Regression layer % required when using trainNetwork
                ];    

            % Set training options
            options = trainingOptions("adam", ...
                ExecutionEnvironment = "cpu", ... % Specify GPU execution
                MaxEpochs = 500, ... % Adjust the number of epochs as needed
                MiniBatchSize = 300, ...
                InitialLearnRate = learn_rate, ... % Use current learning rate
                Shuffle = "every-epoch", ...
                Plots= "training-progress", ...
                Verbose = true, ...
                ValidationData = {val_TDOA_data, val_location_coordinates}, ...
                Metrics = 'rmse', ...
                ValidationPatience = 10);
                
            % Train the network
            loss_function = "mse"; % required when using trainnet
            [net, info] = trainnet(train_TDOA_data, train_location_coordinates, layers, loss_function, options);

            % Extract RMSE values from training information
            RMSE_train = info.TrainingHistory.RMSE(end); % Get final RMSE

            % Extract RMSE values from training information
            RMSE_values(learn_idx, layer_idx, hidden_idx) = info.TrainingHistory.RMSE(end); % Get final RMSE
            hyperparameters(hyperparameters_counter, :) = [learn_rate, layer_size, num_hidden_layer, RMSE_train];

            % Print RMSE for current model
            fprintf('RMSE for current model: %.4f\n', RMSE_values(learn_idx, layer_idx, hidden_idx));

            % Check if current model is better than previous best model
            if RMSE_train < best_RMSE
                best_RMSE = RMSE_train; % Update best RMSE
                best_model = net; % Update best model
                best_info = info;
                best_hyperparameters = struct('LayerSize', layer_size, 'LearnRate', learn_rate, 'HiddenLayer', num_hidden_layer); % Store best hyperparameters
                               
            end

            hyperparameters_counter = hyperparameters_counter + 1;
        end        
    end   
end

% Display best hyperparameters
fprintf('Best hyperparameters:\n');
disp(best_hyperparameters);
fprintf('Best RMSE: %.4f\n', best_RMSE);

% Calculate the duration between the start and end times
EndTime = datetime('now');

durationBetween = EndTime - StartTime;

beep;
%% Save the best model to a file
% save('best_model_33.mat', 'net', 'info')


