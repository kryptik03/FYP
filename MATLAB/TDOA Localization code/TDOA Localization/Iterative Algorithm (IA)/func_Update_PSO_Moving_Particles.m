function func_Update_PSO_Moving_Particles(global_best_position, particles_position, num_particles)

    % Clear previous scatter plot and update the figure properties
    clf;
    axis([0 4 0 11 0 3]);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    grid on;
    hold on;
    view(3); % Set the default 3D view
    
    scatter3(global_best_position(1), global_best_position(2), global_best_position(3), 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
    % Update quiver plot for movement
    h = quiver3(particles_position(:, 1), particles_position(:, 2), particles_position(:, 3), zeros(num_particles, 1), zeros(num_particles, 1), zeros(num_particles, 1));
    % Plot updated positions of particles
    scatter3(particles_position(:, 1), particles_position(:, 2), particles_position(:, 3), 'filled');
    % Pause for a short duration to see the animation
    pause(0.1);

end