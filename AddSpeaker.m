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

file1 = 'C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\56\1730\56-1730-0000.flac';
file2 = 'C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\56\1730\56-1730-0001.flac';
file3 = 'C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\56\1730\56-1730-0002.flac';
file4 = 'C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\56\1730\56-1730-0003.flac';

[audio1, fs1] = audioread(file1);
[audio2, fs2] = audioread(file2);
[audio3, fs3] = audioread(file3);
[audio4, fs4] = audioread(file4);

%% Format data

audio_clips = struct();
audio_clips(1).audio = audio1.';
audio_clips(2).audio = audio2.';
audio_clips(3).audio = audio3.';
audio_clips(4).audio = audio4.';

%% Memorize speaker

speech_processing_model.memorizeSpeaker(audio_clips, fs1, 'Random Internet Reader 2');
save('speech_processing_model.mat', 'speech_processing_model');