clear;
clc;

% TODO:
%   1) Debug stuttering issue when playing filtered sound back
%   2) Create models to evaulate diarization
%   3) Evaluate each model to see benefits vs constraints of each

%% Initialize variables being used
% Load the antialiasing lowpass filter
load('speechFilter.mat');
order = speechFilter.order;

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
[audioWindows,windowSize] = AudioSplitter(audioStream, audioFreq, timeWindowLength, order); % Split the audio file

% Perform bandpass filter on the signals to elimate aliasing
disp('Applying speech (bandpass) filter to audio clips.');

filteredWindows = zeros(length(audioWindows(:, 1)), windowSize - order); % Pre-allocate
for row = 1 : length(audioWindows(:, 1))
    currentFilter = speechFilter.filter(audioWindows(row, :));
    filteredWindows(row, :) = currentFilter(order + 1:end);
end

% Recombine filtered audio into array to test sound
disp('Recombining filtered audio into single vector.');
filteredVector = zeros(length(filteredWindows(:, 1)) * length(filteredWindows(1, :)), 1);
for row = 1 : length(filteredWindows(:, 1))
    for col = 1 : length(filteredWindows(1, :))
        filteredVector(((row - 1) * length(filteredWindows(1, :))) + col) = filteredWindows(row, col);
    end
end

% Calculate the fft
disp('Calculting the FFT of each time window.');
fftWindows = zeros(length(filteredWindows(:, 1)), floor(length(filteredWindows(1,:)) / 2)); % Pre-allocate matrix
fftResolution = audioFreq / windowSize; % Calculate exact fft band gap
fbands = (0 : windowSize - 1) * fftResolution; % Make vector of bands
for row = 1 : length(fftWindows(:, 1)) % Calculate window of each fft
    currentFFT = fft(filteredWindows(row, :));
    fftWindows(row, :) = currentFFT(1: floor(length(currentFFT) / 2));
end

%% Evaluate Models