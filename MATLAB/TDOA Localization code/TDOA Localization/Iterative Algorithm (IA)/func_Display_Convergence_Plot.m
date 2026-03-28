function func_Display_Convergence_Plot(best_iter, best_objective_values)

    % Plotting objective value vs. iteration number
    figure;
    plot((1:best_iter), best_objective_values, 'b.-', 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('Objective Value');
    title('Objective Value Convergence');
    grid on;

end