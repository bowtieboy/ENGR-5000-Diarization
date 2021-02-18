clear;
clc;

% TODO:
%   1) Generate MFCC for time slices
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
timeWindowLength = 1; % Desired difference between freq. bands
disp(['Breaking audio vector into chunks with a time delta of ', num2str(timeWindowLength), ' seconds.']);
[audioWindows,windowSize] = AudioSplitter(audioStream, audioFreq, timeWindowLength, order); % Split the audio file

% Perform bandpass filter on the signals to elimate aliasing
disp('Applying speech (bandpass) filter to audio clips.');
divisions = length(audioWindows(:, 1));
filteredWindowSize = windowSize - order;
filteredWindows = zeros(divisions, filteredWindowSize); % Pre-allocate
for row = 1 : length(audioWindows(:, 1))
    currentFilter = speechFilter.filter(audioWindows(row, :));
    filteredWindows(row, :) = currentFilter(order + 1:end);
end

% Recombine filtered audio into array to test sound
disp('Recombining filtered audio into single vector.');
filteredVector = zeros(divisions * windowSize, 1);
iterator = 1;
for row = 1 : divisions
    for col = 1 : filteredWindowSize
        filteredVector(iterator) = filteredWindows(row, col);
        iterator = iterator + 1;
    end
end

% Seperate out only the spoken audio
disp('Seperating out spoken audio');
spokenData = SpeechSeperator(filteredWindows, audioFreq);
spokenDivisions = length(spokenData);

% Calculate the MFCCs
disp('Calculating the MFCC of each time window.');
for row = 1 : spokenDivisions
    [coeffs, delta, deltaDelta, loc] = mfcc(spokenData(row).speech.', audioFreq);
    spokenData(row).mfccCoeffs = coeffs;
    spokenData(row).mfccDelta = delta;
    spokenData(row).mfccDeltaDelta = deltaDelta;
    spokenData(row).mfccLoc = loc;
end

%% Evaluate Models