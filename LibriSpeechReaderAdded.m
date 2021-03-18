clear;
clc;
%% Get into the correct dir

cd('..\LibriSpeech\train-clean-360');
readerNum = 38;
cd (num2str(readerNum));

%% Loop through files and make audio struct

% Create struct
reader_audio = struct();
index = 1;
total_time = 0;

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
        total_time = total_time + (length(current_audio) / fs);
        index = index + 1;
    end
    cd('..');
end

cd('../../../MATLAB');

% Display the total amount of recorded audio from this user
minutes = floor(total_time / 60);
seconds = mod(total_time, 60);
disp(['Reader #', num2str(readerNum), ' has ', num2str(minutes),...
    ' minutes and ', num2str(seconds), ' seconds of audio data.']);

%% Add reader to model

try
    load('speech_processing_model.mat')
catch
    disp('No SpeechProcessing model was detected, creating a new one');
    speech_processing_model = SpeechProcessing();
end

speech_processing_model.memorizeSpeaker(reader_audio, fs, ['Random Internet Reader #', num2str(readerNum)]);
save('speech_processing_model.mat', 'speech_processing_model');