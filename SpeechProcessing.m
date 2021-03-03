classdef SpeechProcessing < handle
    
    % Private properties
    properties (SetAccess = private)
        bandpass_filter
        d_vectors
    end
    
    % Private methods
    methods (Access = protected)
        
        % Apply the audio pre-processing to the specified entry
        function speechVector = preProcessAudio(obj, audio, fs)
            
            % Apply bandpass filter
            data = obj.bandpass_filter.filter(audio);
            
            % Cut out silences
            speechIdx = detectSpeech(data.', fs);
            speechVector = [];
            for i = 1 : length(speechIdx(:, 1))
                speechVector = [speechVector, data(speechIdx(i, 1) : speechIdx(i, 2))];
            end
                        
        end
        
        % Break the desired audio stream into windows of given size and
        % overlap
        function windows = makeAudioWindows(audio, window_size, window_overlap)
            
            % If overlap is set to window length, change it to 0
            if (window_size <= window_overlap)
                window_overlap = 0;
            end
            
            % Pre-allocate the windows matrix
            window_delta = window_size - window_overlap;
            window_amount = floor(length(audio) / window_delta) - 1;
            windows = zeros(window_amount, window_size);
            
            % Assign the values of the windows matrix
            windows(1, :) = audio(1 : window_size);
            for row = 2 : window_amount
                windows(row, :) = audio(window_delta * (row - 1) : window_size + (window_delta * (row - 1)) - 1);
            end
            
        end
        
        function embeddings = getEmbeddings(audio, fs)
            % Grab audio windows that will be processed
            windows = makeAudioWindows(obj, audio, 1, 0.5);
            
            % Grab embeddings for each window
            embeddings = zeros(length(windows(:, 1), 128));
            for emb = 1 : length(windows(:, 1))
                embeddings(emb, :) = vggishFeatures(windows(emb, :).', fs, 'OverlapPercentage', 0);
            end
        end
        
    end
    
    % Public methods
    methods (Access = public)
        
        % Constructor requires pre-trained bandpass filter. Need to fix
        % this so it is included when the object is created.
        function obj = SpeechProcessing(bandpass_filter)
            obj.bandpass_filter = bandpass_filter;
            addpath('./vggish');
        end
        
        function memorizeSpeaker(obj, speaker_audio, fs)
            
            % Ensure there are at least 4 audio clips
            if (length(speaker_audio) < 4)
                disp('Not enough audio clips in the struct. Need at least 4');
                return;
            end
            
            % Pre-process audio
            processed_audio = struct();
            for clips = 1 : length(speaker_audio)
                processed_audio(clips).audio = preProcessAudio(speaker_audio(clips).audio, fs);
            end
            
            % Calculate speaker embeddings
            
            % Average embeddings
            
            % Store final d-vector
            
        end
        
    end
end