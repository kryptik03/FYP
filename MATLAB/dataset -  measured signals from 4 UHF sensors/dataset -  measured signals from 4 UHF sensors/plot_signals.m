clear all
clc
% Specify the mat file name
location_data = load("location_2.mat");
data = location_data.location;

numChannels = 4;
samplesPerChannel = length(data);
samplingFrequency = 10e9; % Adjust this value based on your oscilloscope settings

timeVector = ((0:samplesPerChannel-1) / samplingFrequency)/ 1e-9;%  %<======== make into ns reference

% ====== Calculate uniform y-axis limits for time-domain plots (normalized all channels) ====== 
y_limits_time = [min(data(:)), max(data(:))];
% =============================================================================================

figure;
for ch = 1:numChannels
    subplot(numChannels,1,ch);
    plot(timeVector, data(:,ch), 'k');
    title(['Sensor ' num2str(ch)], 'FontSize', 12);
    xlabel('Time (ns)', 'FontSize', 12);
    ylabel('Amplitude', 'FontSize', 12);
    ylim(y_limits_time);
end