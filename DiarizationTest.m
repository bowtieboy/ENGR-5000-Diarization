clear;
clc;
%% Initialize variables being used

% If the model exists, load it. Otherwise error out
try
    load('speech_processing_model.mat')
catch
    assert(0, 'No SpeechProcessing model was detected.');
end

file1 = "C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\98\121658\98-121658-0055.flac";
file2 = "C:\Users\froth\Documents\SeniorDesign\Diarization\LibriSpeech\train-clean-360\54\121080\54-121080-0007.flac";
[audio_stream, audio_freq, sepeartion_point] = TwoSpeakerCombiner(file1, file2);

%% Diarization Model

[speakers, probability_matrix, speaker_names] = speech_processing_model.diarizeAudioClip(audio_stream, audio_freq, 0.6);
speech_processing_model.visualizeResults(audio_stream, audio_freq, speakers);
