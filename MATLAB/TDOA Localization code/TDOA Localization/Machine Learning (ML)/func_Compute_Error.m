function [Euclidean_Distance, percentage_error] = func_Compute_Error(true_source_location, estimated_location, receiver1, receiver2, receiver3, receiver4)

    Euclidean_Distance = norm(true_source_location - estimated_location);
    
    Average_distance = (abs(true_source_location - receiver1) + abs(true_source_location - receiver2) ...
        + abs(true_source_location - receiver3) + abs(true_source_location - receiver4))/4;
    
    SUM_TOTAL = norm(Average_distance);
    
    % Calculate the percentage error
    percentage_error = (Euclidean_Distance / SUM_TOTAL) * 100;

end