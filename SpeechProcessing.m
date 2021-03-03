classdef SpeechProcessing < handle
    
    % Private properties
    properties (SetAccess = private)
        bandpass_filter
        d_vectors = struct();
        d_vectors_length = 0;
        embedding_length = 128;
    end
    
    % Private methods
    methods (Access = protected)
        
        % Calculate the cosine similarity between two vectors
        function cos_sim = vectorSimilarity(obj, v1, v2)
            
            cos_sim = dot(v1, v2) / (norm(v1) * norm(v2));
            
        end
        
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
        function windows = makeAudioWindows(obj, audio, fs, window_size, window_overlap)
            
            % If overlap is set to window length, change it to 0
            if (window_size <= window_overlap)
                window_overlap = 0;
            end
            
            % Convert window size and overlamp to sample domain
            window_size = window_size * fs;
            window_overlap = window_overlap * fs;
            
            % Pre-allocate the windows matrix
            window_delta = window_size - window_overlap;
            window_amount = floor(length(audio) / window_delta) - 1;
            
            % Check to make sure the clip is long enough to break into
            % windows. If not, return empty array.
            if (window_amount == 0)
                windows = [];
                return;
            end
            
            windows = zeros(window_amount, window_size);
            
            % Assign the values of the windows matrix
            windows(1, :) = audio(1 : window_size);
            for row = 2 : window_amount
                windows(row, :) = audio(window_delta * (row - 1) : window_size + (window_delta * (row - 1)) - 1);
            end
            
        end
        
        function [embeddings, varargout] = getEmbeddings(obj, audio, fs)
            % Grab audio windows that will be processed
            windows = makeAudioWindows(obj, audio, fs, 1, 0.5);
            
            % If window is empty, return nothing
            if (isempty(windows))
                embeddings = [];
                return;
            end
            
            % Grab embeddings for each window
            embeddings = zeros(length(windows(:, 1)), 128);
            for emb = 1 : length(windows(:, 1))
                embeddings(emb, :) = vggishFeatures(windows(emb, :).', fs, 'OverlapPercentage', 0);
            end
            
            varargout{1} = windows;
            
        end
        
    end
    
    % Public methods
    methods (Access = public)
        
        % Constructor requires pre-trained bandpass filter. Need to fix
        % this so it is included when the object is created.
        function obj = SpeechProcessing()
            
            % Check to see if speechFilter can be loaded
            try
                obj.bandpass_filter = load('speechFilter.mat').speechFilter;
            catch
                assert(0, 'Can not find speechFilter.mat. Is it in this directory?');
            end
            
            % Check to see if the path to vggish exists
            try
                addpath('./vggish');
            catch
                assert(0, 'Can not find path to vggish. Is it in this directory?')
            end
        end
        
        function memorizeSpeaker(obj, speaker_audio, fs, name)
            
            % Ensure there are at least 4 audio clips
            assert(length(speaker_audio) >= 4, 'Not enough audio clips in the struct. Need at least 4.')
            
            % Check if name already exists
            name_exists = 1;
            for names = 1 : obj.d_vectors_length
                if (strcmp(obj.d_vectors(names).name, name))
                    name_exists = 0;
                end
            end
            assert(name_exists, 'User already exists in the database.')
            
            % Pre-process audio
            disp(['Pre-processing ', num2str(length(speaker_audio)) ,' audio clips.']);
            processed_audio = struct();
            for clips = 1 : length(speaker_audio)
                disp(['Processing clip #', num2str(clips)]);
                processed_audio(clips).audio = preProcessAudio(obj, speaker_audio(clips).audio, fs);
            end
            
            % Calculate speaker embeddings
            disp(['Calculating embeddings for ', num2str(length(processed_audio)), ' audio clips.']);
            speaker_embeddings = struct();
            index = 1;
            for speech = 1 : length(processed_audio)
                disp(['Calculating embedding for clip ', num2str(speech), '.']);
                current_embedding = getEmbeddings(obj, processed_audio(speech).audio, fs);
                
                % If the current embedding is empty, do not add it to the
                % struct.
                if (isempty(current_embedding))
                    disp(['Clip ', num2str(speech), ' was not long enough to be diarized.']);
                    continue;
                end
                
                speaker_embeddings(index).emb = current_embedding;
                index = index + 1;
                
            end
            
            % Average embeddings
            disp('Averaging the audio embeddings.');
            average_embedding = zeros(1, obj.embedding_length);
            current_embedding_clip = zeros(1, obj.embedding_length);
            for embedding = 1 : length(speaker_embeddings)
                for row = 1 : length(speaker_embeddings(embedding).emb(:, 1))
                    current_embedding_clip = average_embedding + speaker_embeddings(embedding).emb(row, :);
                end
                current_embedding_clip = current_embedding_clip ./ length(speaker_embeddings(embedding).emb(:, 1));
                average_embedding = average_embedding + current_embedding_clip;
            end
            average_embedding = average_embedding ./ length(speaker_embeddings);
            
            % Store final d-vector
            disp('Storing the final embedding.');
            obj.d_vectors_length = obj.d_vectors_length + 1;
            obj.d_vectors(obj.d_vectors_length).speaker_embedding = average_embedding;
            obj.d_vectors(obj.d_vectors_length).name = name;
        end
        
        % Diarize the given audio clip
        function [speakers, similarities] = diarizeAudioClip(obj, audio, fs, threshold)
            
            % Make sure audio clip is long enough, otherwise error out
            assert((length(audio) / fs) >= 1, 'Audio clip is not long enough to diarize.');
            
            % Pre-process audio
            disp('Pre-processing audio clip.');
            processed_audio = preProcessAudio(obj, audio, fs);
            
            % Calculate speaker embeddings
            disp('Extracting d-vectors from audio clip.');
            speaker_embeddings = getEmbeddings(obj, processed_audio, fs);
            
            % Determine the similarity
            disp('Calculating the similarities between the d-vectors and recorded speakers.');
            speakers = strings(1, length(speaker_embeddings(:, 1)));
            similarities = zeros(obj.d_vectors_length, length(speakers));
            for w = 1 : length(speakers)
                % Compare the windows embedding vs the recorded ones
                for dvec = 1 : obj.d_vectors_length
                    similarities(dvec, w) = vectorSimilarity(obj, speaker_embeddings(w, :), obj.d_vectors(dvec).speaker_embedding);
                end
            end
            
            % Mark the speaker with the highest similarity as the person in
            % the window
            for s = 1 : length(speakers)
                [M,I] = max(similarities(:, s));
                
                if(M < threshold)
                    speakers(s) = "Uncrecognized speaker";
                    continue;
                end
                
                speakers(s) = obj.d_vectors(I).name;
            end
        end
    end
end