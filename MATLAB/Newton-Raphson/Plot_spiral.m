% Define parameter range
a = linspace(0, 100, 10000);   % use many points for smooth curve

% Define parametric equations
x = a .* cos(a);
y = a .* sin(a);

% Plot
figure;
plot(x, y, 'b');     % 'b' for blue line
axis equal;          % keep aspect ratio equal
grid on;
xlabel('x = a cos(a)');
ylabel('y = a sin(a)');
title('Parametric Plot of x = a cos(a), y = a sin(a)');