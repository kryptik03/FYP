clear
clc

%% 1. Call Sensors_Data - Contained: 4 sensor coordinates, Speed of EM wave, PD TDOA & PD coordinates (for benchmarking).
[receiver1, receiver2, receiver3, receiver4, speed_of_EM, data_predict] = func_Sensors_Data();

% Loop through all the TDOA in data_predict
num_of_loop = size(data_predict, 1);

Computation_Time = zeros(num_of_loop, 1);
FOR_EXCEL = cell(num_of_loop, 5);


% trial

for data_set = 1:num_of_loop
    StartTime = datetime('now');

    %% 3. Perform Sorting_data_predict: sorting the respective PD TDOA and coordinate in the data_predict list
    [true_source_location, TDOA_N] = func_Sorting_data_predict(data_predict, data_set);

    % PSO START HERE:
    
    % Parameters for PSO and convergence criterion
    max_iterations_pso = 50000;
    convergence_threshold = 1e-12; % Threshold for change in global best value
    convergence_iter_threshold = 1000; % Number of consecutive iterations with no improvement to trigger convergence

    convergence_counter = 0; % Initialize convergence counter
    
    global_best_value = inf;    prev_global_best_value = 0;    objective_values = zeros; % To clear current data_set and ready for next data_set

    % Initialization
    num_particles = 1000;
    inertia_weight = 0.1;
    c1 = 3.0; % Cognitive coefficient
    c2 = 1.0; % Social coefficient

    % Initialize particles' positions and velocities *within boundary
    particles_position = zeros(num_particles, 3);    particles_velocity = zeros(num_particles, 3);    
    for i = 1:num_particles
        % particles_position(i, 1) = rand() * 26 - 15; % num_particles(:, 1) range from -15 to 11
        particles_position(i, 1) = rand() * 11 + 0.5; % num_particles(:, 1) range from 1 to 11
        particles_position(i, 2) = rand() *11 + 0.5; % num_particles(:, 2) range between 1 to 11
        particles_position(i, 3) = rand() * 6 + 0.1; % num_particles(:, 3) range between 0.3 to 2
        
        particles_velocity(i, :) = rand(1, 3); % Initialize velocities randomly
    end

    objective_values_particles = zeros(num_particles, 1);

    % Initialize personal best positions and the corresponding objective values
    personal_best_positions = particles_position;    personal_best_values = ones(num_particles, 1);
    
    % Initialize the global best position and the corresponding objective value
    global_best_position = particles_position(1, :);  

    % Initialize the plot -> For Plotting Live Objective Value
    % figure;
    % h = plot(NaN, NaN, 'LineWidth', 2);
    % xlabel('Iterations');
    % ylabel('Global Best Value');
    % title('Global Best Value vs Iteration');
    % grid on;

    %% 9. Display_PSO_Moving_Particles
    % func_Display_PSO_Moving_Particles(global_best_position, particles_position, num_particles);

    for iter_pso = 1:max_iterations_pso
    
        % Evaluate the FITNESS of each particle
        for i = 1:num_particles
            x0 = particles_position(i, 1);
            y0 = particles_position(i, 2);
            z0 = particles_position(i, 3);

            %% 4. Objective_Function: Formulate of c = d/t
            [r1, r2, r3] = func_Objective_Function(x0, y0, z0, receiver1, receiver2, receiver3, receiver4, speed_of_EM, TDOA_N);
                                  
            % Calculate the objective value (sum of squared residuals)
            objective_values_particles(i) = sqrt((r1^2 + r2^2 + r3^2)/3); % RMSE

            % Update personal best if the current position is better
            if objective_values_particles(i) < personal_best_values(i)
                personal_best_positions(i, :) = particles_position(i, :);
                personal_best_values(i) = objective_values_particles(i);
            end
            
            % Update global best if the current position is better
            if objective_values_particles(i) < global_best_value
                global_best_position = particles_position(i, :);
                global_best_value = objective_values_particles(i);
            end
             
            % Update velocity
            particles_velocity(i, :) = inertia_weight * particles_velocity(i, :) + ...
                c1 * rand() * (personal_best_positions(i, :) - particles_position(i, :)) + ...
                c2 * rand() * (global_best_position - particles_position(i, :));
            
            % Update position
            particles_position(i, :) = particles_position(i, :) + particles_velocity(i, :);

            % Bound positions within search space [1, 1, 0] (lower bounds) + [11, 11, 1.8] (upper bounds)
            particles_position(i, :) = max(min(particles_position(i, :), [11 11 6]), [0.5 0.5 0.1]); %<=======================
            % particles_position(i, :) = max(min(particles_position(i, :), [11 11 2]), [-15 1 0.3]);

            objective_values(iter_pso, 1) = global_best_value;
        end        
        %% 5. Convergence_check: stop running when convergence met
        [convergence_counter] = func_Convergence_Check(iter_pso, global_best_value, prev_global_best_value, convergence_threshold, convergence_counter);
               
        % Check convergence criterion
        if convergence_counter >= convergence_iter_threshold
            fprintf('Convergence did not improve after %d iterations (iter_pso = %d).\n', convergence_iter_threshold, iter_pso);
            break; % Exit the PSO loop if convergence criterion is met
        end               
        
        % Store current global best value for next iteration
        prev_global_best_value = global_best_value;

        % %% For Plotting Live Objective Value
        % % Update the plot with the new data
        % set(h, 'XData', 1:iter_pso, 'YData', objective_values(1:iter_pso));
        % 
        % % Pause to allow the plot to update
        % pause(0.1); % Adjust pause duration as needed

        %% 10. Update_PSO_Moving_Particles
        % func_Update_PSO_Moving_Particles(global_best_position, particles_position, num_particles)

    end    

    % Final estimated source location from PSO
    estimated_location = global_best_position;   
    
    %% 6. Compute_Error
    [Euclidean_Distance, percentage_error] = func_Compute_Error(true_source_location, estimated_location, receiver1, receiver2, receiver3, receiver4);
    
    %% 7. Display_Convergence_Plot
    % func_Display_Convergence_Plot(iter_pso, objective_values);
    
    %% 8. Display_results
    func_Display_Results(estimated_location, true_source_location, Euclidean_Distance, percentage_error);

    EndTime = datetime('now');

    durationBetween = EndTime - StartTime;
    
    % Convert the duration to seconds before assigning
    % Computation_Time(data_set, 1) = seconds(durationBetween);

    % Round off for coordinate only before copied to excel
    % estimated_location = arrayfun(@(x) sprintf('%.2f', x), estimated_location, 'UniformOutput', false);
    true_source_location = arrayfun(@(x) sprintf('%.2f', x), true_source_location, 'UniformOutput', false);

    % Format Numbers as Strings Before Export
    % With round off
    % Round to 4 significant figures
    FOR_EXCEL{data_set, 1} = round(percentage_error, 2);
    FOR_EXCEL{data_set, 2} = round(Euclidean_Distance, 3);
    FOR_EXCEL{data_set, 3} = round(seconds(durationBetween), 2);
    FOR_EXCEL{data_set, 4} = round(estimated_location(1), 2);
    FOR_EXCEL{data_set, 5} = round(estimated_location(2), 2);
    FOR_EXCEL{data_set, 6} = round(estimated_location(3), 2);
    FOR_EXCEL{data_set, 7} = strjoin(true_source_location, ', ');
end

% Save the data
save('FOR_EXCEL_PSO.mat', 'FOR_EXCEL');

writecell(FOR_EXCEL, 'converted_data.xlsx');