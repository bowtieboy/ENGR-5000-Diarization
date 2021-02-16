clear;
clc;

% TODO:
%   1) Create models to evaulate diarization
%   2) Evaluate each model to see benefits vs constraints of each

%% Initialize variables being used
% Load the antialiasing lowpass filter
load('speechFilter.mat');

% Directory variables
baseDir = "../Conversational Files/"; % Directory where the audio clips are
audioFile = "breakfast1.m4a"; % Which file to grab

% Open the audio file
disp('Opening audio clip: ' + audioFile);
[audioStream,audioFreq] = audioread(baseDir + audioFile);

%% Preparation Calculations

% Create matrix of audio values
timeWindowLength = 0.1; % Desired difference between freq. bands
disp(['Breaking audio vector into chunks with a time delta of ', num2str(timeWindowLength), ' seconds.']);
[audioWindows,windowSize] = AudioSplitter(audioStream, audioFreq, timeWindowLength); % Split the audio file

% Perform bandpass filter on the signals to elimate aliasing
disp('Applying speech (bandpass) filter to audio clips.');
order = speechFilter.order;
filteredWindows = zeros(length(audioWindows), windowSize - order); % Pre-allocate
for row = 1 : length(audioWindows)
    currentFilter = speechFilter.filter(audioWindows(row, :));
    filteredWindows(row, :) = currentFilter(order + 1:end);
end

% Calculate the fft
disp('Calculting the FFT of each time window.');
fftWindows = zeros(length(filteredWindows), floor(length(filteredWindows(1,:)) / 2)); % Pre-allocate matrix
fftResolution = audioFreq / windowSize; % Calculate exact fft band gap
fbands = (0 : windowSize - 1) * fftResolution; % Make vector of bands
for row = 1 : length(fftWindows) % Calculate window of each fft
    currentFFT = fft(filteredWindows(row, :));
    fftWindows(row, :) = currentFFT(1: floor(length(currentFFT) / 2));
end