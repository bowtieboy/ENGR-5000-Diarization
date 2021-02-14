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
desiredFFTResolution = 10; % Desired difference between freq. bands
windowSize = floor(audioFreq / desiredFFTResolution); % Get samples / window
audioWindows = AudioSplitter(audioStream, windowSize); % Split the audio file

% Calculate the fft
fftWindows = zeros(length(audioWindows), length(audioWindows(1,:))); % Pre-allocate matrix
actualFFTResolution = audioFreq / windowSize; % Calculate exact fft band gap
fbands = (0 : windowSize - 1) * actualFFTResolution; % Make vector of bands
for row = 1 : length(fftWindows) % Calculate window of each fft
    fftWindows(row, :) = fft(audioWindows(row, :));
end