function func_plot_digital_twin_signal(t, s)
    
    % Plotting
    figure;
    plot(t * 1e9, s, 'b', 'LineWidth', 1.5);
    xlabel('Time (ns)', 'FontSize', 12);
    ylabel('Voltage', 'FontSize', 12);
    title('Simulated Partial Discharge (PD) Signal', 'FontSize', 12);
    grid on;

    % Customize axes
    ax = gca;
    ax.FontSize = 16; % Adjust tick label font size

end