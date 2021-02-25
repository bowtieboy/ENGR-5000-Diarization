clear;
clc;

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

% % Apply a noise gate to the audio
% disp('Applying noise gate to audio clip.');
 gatedAudio = audioWindows;
% dRG = noiseGate(-75, 'SampleRate', audioFreq);
% for row = 1 : length(audioWindows(:, 1))
%     gatedAudio(row, :) = dRG(audioWindows(row, :).').';
% end

% Perform bandpass filter on the signals to elimate aliasing
disp('Applying speech (bandpass) filter to audio clips.');
divisions = length(gatedAudio(:, 1));
filteredWindowSize = windowSize - order;
filteredWindows = zeros(divisions, filteredWindowSize); % Pre-allocate
for row = 1 : length(gatedAudio(:, 1))
    currentFilter = speechFilter.filter(gatedAudio(row, :));
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

% Calculate the FFTs
disp('Calculating FFT of spoken data windows');
for row = 1 : spokenDivisions
    tempFFT = fft(spokenData(row).speech);
    spokenData(row).fft = tempFFT(1 : floor(length(tempFFT) / 2));
    binRes = audioFreq / length(spokenData(row).speech);
    tempBins = (0 : length(spokenData(row).speech) - 1) * binRes;
    spokenData(row).fftBins = tempBins(1 : floor(length(tempBins) / 2));
    spokenData(row).binRes = binRes;
end

% Calculate the MFCCs
disp('Calculating the MFCC of each time window.');
for row = 1 : spokenDivisions
    [coeffs, delta, deltaDelta, loc] = mfcc(spokenData(row).speech.', audioFreq);
    spokenData(row).mfccCoeffs = coeffs;
    spokenData(row).mfccDelta = delta;
    spokenData(row).mfccDeltaDelta = deltaDelta;
    spokenData(row).mfccLoc = loc;
end

%% Diarization Model

% Step 1: Change of speaker detection
% Step 2: Identify number of unique speakers in clip
% Step 3: Clustering algorithm (GMM?)
% Step 4: Seperate audio of individual speakers
