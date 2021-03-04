clear;
clc;
%% Get into the correct dir

cd('..\LibriSpeech\train-clean-360');
readerNum = 7558;
cd (num2str(readerNum));

%% Loop through files and make audio struct

% Create struct
reader_audio = struct();
index = 1;

% Get chapters subfolders
disp('Creating audio struct.');
chapters = dir();
for c = 3 : length(chapters)
    cd(chapters(c).name);
    % Get recordings in chapter
    recordings = dir();
    for r = 3 : length(recordings)
        % If file is .txt, ignore it
        if(contains(recordings(r).name, '.txt'))
            continue
        end
        [current_audio, fs] = audioread(recordings(r).name);
        reader_audio(index).audio = current_audio.';
        index = index + 1;
    end
    cd('..');
end

cd('../../../MATLAB');

%% Add reader to model

try
    load('speech_processing_model.mat')
catch
    disp('No SpeechProcessing model was detected, creating a new one');
    speech_processing_model = SpeechProcessing();
end

speech_processing_model.memorizeSpeaker(reader_audio, fs, ['Random Internet Reader #', num2str(readerNum)]);
save('speech_processing_model.mat', 'speech_processing_model');