function [convergence_counter] = func_Convergence_Check(iter, global_best_value, prev_global_best_value, convergence_threshold, convergence_counter)
            
    % Check for convergence based on change in global best value
    if iter > 1 && abs(global_best_value - prev_global_best_value) < convergence_threshold
        convergence_counter = convergence_counter + 1;
    else
        convergence_counter = 0; % Reset convergence counter if there's improvement
    end
    
end