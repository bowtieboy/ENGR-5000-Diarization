% function [indices, numOfSpeakers] = AudioDiarization(speech, sampleRate, windowSize, windowOverlap)
%     
%     samplesPerWindow = round(sampleRate * windowSize);
%     window = hann(samplesPerWindow, 'periodic');
%     overlapLength = round(sampleRate * windowOverlap);
%     
%     % Preparation calculations
%     speech = normalize(speech);
%     [coeffs, delta, deltaDelta, loc] = mfcc(speech, sampleRate, 'Window', window, 'OverlapLength', overlapLength);
%     pitches = pitch(speech, sampleRate, 'WindowLength', length(window), 'OverlapLength', overlapLength);
%     deltaPitches = zeros(1, length(pitches));
%     for i = 2 : length(pitches)
%         deltaPitches(i) = pitches(i) - pitches(i - 1);
%     end
%     
%     numOfSpeakers = 2;
%     gmmModelCoeffs = fitgmdist(coeffs, numOfSpeakers, 'CovarianceType', 'diagonal');
%     gmmModelPitches = fitgmdist(pitches, numOfSpeakers, 'CovarianceType', 'diagonal');
%     indicesCoeffs = cluster(gmmModelCoeffs, coeffs);
%     indicesPitches = cluster(gmmModelPitches, pitches);
%     
%     indices = [indicesCoeffs, indicesPitches];
% end

classdef AudioDiarization < handle
    
    properties (SetAccess = private)
        bandpass_filter
        d_vectors
        network_params
        modified_audio_data = struct();
        modified_data_entries = 0;
    end
    
    properties (SetAccess = public)
        raw_audio_data = struct();
        data_entries = 0;
    end
    
    methods (Access = protected)
        
        % Apply the audio pre-processing to the specified entry
        function preProcessAudio(obj, data_entry)
            % Check to make sure data_entry has data in the struct
            if (data_entry > obj.data_entries)
                disp(['Data entry ', num2str(data_entry), ' is not in the data set.']);
                return;
            end
            
            data = obj.raw_audio_data(data_entry).audio;
            % Apply the bandpass filter
            data = obj.bandpass_filter.filter(data);
            
            % Cut out silences
            speechIdx = detectSpeech(data.', obj.raw_audio_data(data_entry).fs);
            speechVector = [];
            for i = 1 : length(speechIdx(:, 1))
                speechVector = [speechVector, data(speechIdx(i, 1) : speechIdx(i, 2))];
            end
            
            % Add data 
            obj.modified_data_entries = obj.modified_data_entries + 1;
            obj.modified_audio_data(obj.modified_data_entries).audio = speechVector;
            obj.modified_audio_data(obj.modified_data_entries).fs = obj.raw_audio_data(data_entry).fs;
            
        end
        
        function windows = makeAudioWindows(obj, data_entry, window_size, window_overlap)
            
            % If overlap is set to window length, change it to 0
            if (window_size == window_overlap)
                window_overlap = 0;
            end
            
            % Pre-allocate the windows matrix
            current_data = obj.modified_audio_data(data_entry).audio;
            window_delta = window_size - window_overlap;
            window_amount = floor(length(current_data) / window_delta) - 1;
            windows = zeros(window_amount, window_size);
            
            % Assign the values of the windows matrix
            windows(1, :) = current_data(1 : window_size);
            for row = 2 : window_amount
                windows(row, :) = current_data(window_delta * (row - 1) : window_size + (window_delta * (row - 1)) - 1);
            end
            
        end
        
    end
    
    methods (Access = public)
        
        % Constructor requires pre-trained bandpass filter. Need to fix
        % this so it is included when the object is created.
        function obj = AudioDiarization(bandpass_filter)
            obj.bandpass_filter = bandpass_filter;
        end
               
        % Add new audio clip to the object's database
        function addAudioClip(obj, newAudio, sampleFreq)
            obj.data_entries = obj.data_entries + 1;
            obj.raw_audio_data(obj.data_entries).audio = newAudio;
            obj.raw_audio_data(obj.data_entries).fs = sampleFreq;
            
            
            preProcessAudio(obj, obj.data_entries);
            
        end
        
        % Diarize the audio clip with the given window time and overlap
        function diarizeAudio(obj, data_entry, window_time, window_overlap_time)
            
            % Grab audio windows that will be processed
            window_size = window_time * obj.modified_audio_data(data_entry).fs;
            window_overlap = window_overlap_time * obj.modified_audio_data(data_entry).fs;
            windows = makeAudioWindows(obj, data_entry, window_size, window_overlap);
        end
        
    end
end