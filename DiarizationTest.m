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
% [audio, fs, sepeartion_point] = TwoSpeakerCombiner(file1, file2);

% Real speakers
file = "C:\Users\froth\Documents\SeniorDesign\Diarization\Real Speakers\Evaluation\mel_matt_mel.mp3";
[audio, fs] = audioread(file);
audio = audio.';

%% Diarization Model

% Create annotations
annotated_speakers = speech_processing_model.annotateAudio(audio, fs, 0.5);

% Visualize diarization
speech_processing_model.visualizeResults(audio, fs, annotated_speakers);

% Print text to screen
speech_list = speech_processing_model.printAnnontation(annotated_speakers);
