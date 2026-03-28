function func_Display_PSO_Moving_Particles(global_best_position, particles_position, num_particles)

    % Initialize figure
    figure;
    axis([0 4 0 11 0 3]);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    grid on;
    hold on;
    view(3); % Set the default 3D view
     
    % Plot searched point
    scatter3(global_best_position(1), global_best_position(2), global_best_position(3), 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
    % Initialize quiver plot for movement
    h = quiver3(particles_position(:, 1), particles_position(:, 2), particles_position(:, 3), zeros(num_particles, 1), zeros(num_particles, 1), zeros(num_particles, 1));

end