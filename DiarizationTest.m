clear;
clc;
%% Initialize variables being used

% If the model exists, load it. Otherwise error out
try
    load('speech_processing_model.mat')
catch
    assert(0, 'No SpeechProcessing model was detected.');
end

% Librispeech
% file1 = "C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\38\121024\38-121024-0000.flac";
% file2 = "C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\98\121658\98-121658-0004.flac";
% [audio_stream, audio_freq, sepeartion_point] = TwoSpeakerCombiner(file1, file2);

% Real speakers
file = "C:\Users\froth\Documents\SeniorDesign\Diarization\Real Speakers\Evaluation\matt1.mp3";
[audio_stream, audio_freq] = audioread(file);
audio_stream = audio_stream.';

%% Diarization Model

[speakers, probability_matrix, speaker_names] = speech_processing_model.diarizeAudioClip(audio_stream, audio_freq, 0.6);
speech_processing_model.visualizeResults(audio_stream, audio_freq, speakers);
