clear;
clc;
%% Test output

% Load scripts to the path
addpath('./speech2text');

% Load audio clip
file = "C:\Users\froth\Documents\SeniorDesign\Diarization\Real Speakers\Evaluation\matt1.mp3";
[audio, fs] = audioread(file);

% Create speech object
speechObject = speechClient('Google','languageCode','en-US',...
    'sampleRateHertz',16000,'enableWordTimeOffsets',true);

% Perform calculation
tableOut = speech2text(speechObject,audio,fs);