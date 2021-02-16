clear;
clc;
%% Initialize variables being used

% Directory variables
baseDir = "../Conversational Files/"; % Directory where the audio clips are
audioFile = "breakfast1.m4a"; % Which file to grab

% Open the audio file
[audioStream,audioFreq] = audioread(baseDir + audioFile);

%% Calculations

% Create matrix of audio values
timeWindowLength = 0.1; % Desired difference between freq. bands
[audioWindows,windowSize] = AudioSplitter(audioStream, audioFreq, timeWindowLength); % Split the audio file


% Calculate the fft
fftWindows = zeros(length(audioWindows), length(audioWindows(1,:)) / 2); % Pre-allocate matrix
actualFFTResolution = audioFreq / windowSize; % Calculate exact fft band gap
fbands = (0 : windowSize - 1) * actualFFTResolution; % Make vector of bands
for row = 1 : length(fftWindows) % Calculate window of each fft
    currentFFT = fft(audioWindows(row, :));
    fftWindows(row, :) = currentFFT(1: length(currentFFT) / 2);
end