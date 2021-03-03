clear;
clc;

%% Initialize variables being used
% Load the antialiasing lowpass filter
load('speechFilter.mat');
order = speechFilter.order;

% % Directory variables
% baseDir = "../Conversational Files/"; % Directory where the audio clips are
% audioFile = "breakfast1.m4a"; % Which file to grab
% 
% % Open the audio file
% disp('Opening audio clip: ' + audioFile);
% [audioStream,audioFreq] = audioread(baseDir + audioFile);
file1 = "C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\98\121658\98-121658-0059.flac";
file2 = "C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\54\121080\54-121080-0007.flac";
[audioStream, audioFreq] = TwoSpeakerCombiner(file1, file2);

%% Diarization Model

da = SpeechProcessing(speechFilter);
da.diarizeAudio(audioStream, audioFreq, 1, 0.5);