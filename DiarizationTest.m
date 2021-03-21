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
file = "C:\Users\froth\Documents\SeniorDesign\Diarization\Real Speakers\Evaluation\matt_and_mel.mp3";
[audio_stream, audio_freq] = audioread(file);
audio_stream = audio_stream.';

%% Diarization Model

% Create annotations
[annotated_speakers, speakers] = speech_processing_model.annotateAudio(audio_stream, audio_freq, 0.6);
% Visualize diarization
speech_processing_model.visualizeResults(audio_stream, audio_freq, speakers);


