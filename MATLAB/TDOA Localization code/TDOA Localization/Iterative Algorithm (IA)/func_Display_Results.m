function func_Display_Results(best_estimated_location, best_true_source_location, best_Euclidean_Distance, best_percentage_error)

    fprintf('Estimated source location: (%.2f, %.2f, %.2f)\n', best_estimated_location);
    fprintf('True source location: (%.2f, %.2f, %.2f)\n', best_true_source_location);
    fprintf('Euclidean Distance: %.2f meter\n', best_Euclidean_Distance);
    fprintf('Percentage error: %.2f%%\n\n', best_percentage_error);

end