clear;
clc;
%% Check if SpeechProcessing model exists

try
    load('speech_processing_model.mat')
catch
    disp('No SpeechProcessing model was detected, creating a new one');
    speech_processing_model = SpeechProcessing();
end
%% Get audio data

file = 'C:\Users\froth\Documents\SeniorDesign\Diarization\Real Speakers\Matt\genesis 1,1-15.flac';
[audio, fs] = audioread(file);
name = 'Matt Lima';

% Display the total amount of recorded audio from this user
total_time = length(audio) / fs;
minutes = floor(total_time / 60);
seconds = mod(total_time, 60);
disp(['Reader: ', name, ' has ', num2str(minutes),...
    ' minutes and ', num2str(seconds), ' seconds of audio data.']);
%% Format data

audio_clips = struct();
audio_clips(1).audio = audio.';
%% Memorize speaker

speech_processing_model.memorizeSpeaker(audio_clips, fs, name);
save('speech_processing_model.mat', 'speech_processing_model');