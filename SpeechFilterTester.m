clear;
clc;
%% Setup initial variables
load('speech_filter.mat');

fs = 48000;
endTime = 1;
freq1 = 60;
freq2 = 2000;
freq3 = 15000; % Needs to be larger freq so plotting doesnt look bad

t = linspace(0, endTime, fs*endTime);

y1 = 2 .* sin(2 * pi * freq1 .* t);
y2 = sin(2 * pi * freq2 .* t);
y3 = 0.5 .* sin(2 * pi * freq3 .* t);

y4 = y1 + y2 + y3;

dataLength = length(y3);
%% Perform Calculations
filteredData = speechFilter.filter(y4);
fbands = (0 : dataLength - 1) * (fs / dataLength);

freqDomainData = fft(y4);
freqDomainFilteredData = fft(filteredData);

freqDomainData = abs(freqDomainData) .^ 2;
freqDomainFilteredData = abs(freqDomainFilteredData) .^ 2;

%% Plot results
fig1 = figure();
subplot(2,1,1);
semilogx(fbands(1 : dataLength / 2), freqDomainData(1 : dataLength / 2), 'r');
xlabel('Frequency (Hz)');
ylabel('Power (W)');
subplot(2,1,2);
semilogx(fbands(1 : dataLength / 2), freqDomainFilteredData(1 : dataLength / 2), 'b');
xlabel('Frequency (Hz)');
ylabel('Power (W)');

order = speechFilter.order;
fig2 = figure();
subplot(2,1,1);
plot(t(order : order + 100), y4(order : order + 100), 'r');
xlabel('Time (s)');
ylabel('Signal (V)');
subplot(2,1,2);
plot(t(order : order + 100), filteredData(order : order + 100), 'b');
xlabel('Time (s)');
ylabel('Signal (V)');