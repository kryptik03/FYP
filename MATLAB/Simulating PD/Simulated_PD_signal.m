clear
clc

% Parameters
A = 1;              % Amplitude
alpha = 5e8;        % Damping factor [1/s]
f = 800e6;          % Frequency [Hz]
tdelay = 20.1e-9;   % Delay time [s] (e.g., 20 ns)

Fs = 10e9;          % Sampling frequency [Hz]
T = 100e-9;         % Total time duration [s]

t = 0 : 1/Fs : T; % Time vector

% PD signal model
s = zeros(size(t)); % Pre-allocate
idx = t >= tdelay;  % Start after delay
s(idx) = A * exp(-alpha * (t(idx)-tdelay)) .* sin(2*pi*f*(t(idx)-tdelay));

func_plot_digital_twin_signal(t, s);