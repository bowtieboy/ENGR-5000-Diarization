classdef (ConstructOnLoad) SpeechProcessing < handle
    
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
        function [speech_vector, speech_indices] = preProcessAudio(obj, audio, fs)
            
            % Apply bandpass filter
            data = obj.bandpass_filter.filter(audio);
            
            % Cut out silences
            speechIdx = detectSpeech(data.', fs);
            speech_vector = [];
            for i = 1 : length(speechIdx(:, 1))
                speech_vector = [speech_vector, data(speechIdx(i, 1) : speechIdx(i, 2))];
            end
            
            if (nargout > 1)
                speech_indices = speechIdx;
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
        
        function embeddings = getEmbeddings(obj, audio, fs)
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
        end
        
        function speaker_indices = getSpeechSpeakerIndices(obj, fs, windows, window_size, window_overlap, speakers)
                        
            % Convert window size and overlamp to sample domain
            window_size = window_size * fs;
            window_overlap = window_overlap * fs;
            window_delta = window_size - window_overlap;
            
            % Find the indices attributed to each window
            window_indices = zeros(length(windows(:, 1)), 2);
            window_indices(1, :) = [1, length(windows(1, :))];
            for w = 2 : length(windows(:, 1))
                window_indices(w, :) = [window_delta * (w - 1), window_size + (window_delta * (w - 1)) - 1];
            end
            
            % Attribute the indices to speakers
            speaker_indices = struct();
            speaker_changes = 1;
            current_speaker = speakers(1);
            start_idx = 1;
            for s = 2 : length(window_indices(:, 1))
                % If speaker has not changed, do nothing
                if (strcmp(speakers(s), current_speaker))
                    continue;
                end
                
                % Use the end of the last window as the final idx for the
                % speaker and increment the speaker changes
                speaker_indices(speaker_changes).idx = [start_idx, window_indices(s - 1, 2)];
                speaker_indices(speaker_changes).speaker = current_speaker;
                current_speaker = speakers(s);
                start_idx = window_indices(s, 1);
                speaker_changes = speaker_changes + 1;
            end
        end
        
        function original_speaker_indices = getOriginalSpeakerIndices(obj, speech_indices, speech_speaker_indices)
            
            % Initial values
            original_speaker_indices = struct();
            speaker_change = 1;
            leftover_samples = 0;
            for i = 1 : length(speech_indices(:, 1))
                
                % Define the new start point
                start_idx = speech_indices(i, 1);
                
                % Calculate the difference between the currnt point and the
                % end of the current speakers segment
                delta_idx = speech_indices(i, 2) - start_idx;
                
                % If there are sample points left for this speaker that fit
                % within the current speach window, append them and
                % continue on
                if ((leftover_samples > 0) && (delta_idx > leftover_samples))
                    original_speaker_indices(speaker_change).idx = [original_speaker_indices(speaker_change).idx; [start_idx, start_idx + leftover_samples]];
                    speaker_change = speaker_change + 1;
                    % If all the speakers have been calculated, exit the
                    % function
                    if (speaker_change > length(speech_speaker_indices))
                        return;
                    end
                    start_idx = start_idx + leftover_samples - (speech_speaker_indices(speaker_change - 1).idx(2) - speech_speaker_indices(speaker_change).idx(1));
                    delta_idx = speech_indices(i, 2) - start_idx;
                    leftover_samples = 0;
                end
                
                % If there are sample points left for this speaker that
                % dont fit within the current speach window, append them to
                % the current speaker and continue on
                if (leftover_samples > 0)
                    original_speaker_indices(speaker_change).idx = [original_speaker_indices(speaker_change).idx ;[start_idx, speech_indices(i, 2)]];
                    leftover_samples = leftover_samples - delta_idx;
                    continue;
                end
                             
                % If the number of samples left in the speach window is
                % less than the samples of the current speaker speaking,
                % map all of their points to the original vector and adjust
                % the starting position. Repeat until the speech window is
                % smaller than the next speakers number of samples
                while ((delta_idx > (speech_speaker_indices(speaker_change).idx(2) - speech_speaker_indices(speaker_change).idx(1))) && (leftover_samples == 0))
                    original_speaker_indices(speaker_change).idx = [start_idx, start_idx + (speech_speaker_indices(speaker_change).idx(2) - speech_speaker_indices(speaker_change).idx(1))];
                    original_speaker_indices(speaker_change).speaker = speech_speaker_indices(speaker_change).speaker;
                    speaker_change = speaker_change + 1;
                    % If all the speakers have been calculated, exit the
                    % function
                    if (speaker_change > length(speech_speaker_indices))
                        return;
                    end
                    start_idx = start_idx + (speech_speaker_indices(speaker_change).idx(2) - speech_speaker_indices(speaker_change).idx(1)) - (speech_speaker_indices(speaker_change - 1).idx(2) - speech_speaker_indices(speaker_change).idx(1)) + 1;
                    delta_idx = speech_indices(i, 2) - start_idx;
                end
                
                % Once there are more points in the speaker window then
                % there are in the speech window, set the indices for the
                % current speaker then continue through the loop
                original_speaker_indices(speaker_change).idx = [start_idx, speech_indices(i, 2)];
                original_speaker_indices(speaker_change).speaker = speech_speaker_indices(speaker_change).speaker;
                leftover_samples = (speech_speaker_indices(speaker_change).idx(2) - speech_speaker_indices(speaker_change).idx(1)) - delta_idx;
                
            end
            
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
        function [original_speaker_indices, return_similarities] = diarizeAudioClip(obj, audio, fs, threshold)
            
            % Make sure audio clip is long enough, otherwise error out
            assert((length(audio) / fs) >= 1, 'Audio clip is not long enough to diarize.');
            
            % Pre-process audio
            disp('Pre-processing audio clip.');
            [processed_audio, speech_indices] = preProcessAudio(obj, audio, fs);
            
            % Calculate speaker embeddings
            disp('Extracting d-vectors from audio clip.');
            speaker_embeddings = getEmbeddings(obj, processed_audio, fs);
            
            % Determine the similarity
            disp('Calculating the similarities between the d-vectors and recorded speakers.');
            speakers = strings(length(speaker_embeddings(:, 1)), 1);
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
            
            % Get the indices of the original audio clip that each speaker
            % is attributed to
            windows = makeAudioWindows(obj, processed_audio, fs, 1, 0.5);
            speech_speaker_indices = getSpeechSpeakerIndices(obj, fs, windows, 1, 0.5, speakers);
            original_speaker_indices = getOriginalSpeakerIndices(obj, speech_indices, speech_speaker_indices);
            
            % Return similarities if requested
            if (nargout > 1)
                return_similarities = similarities;
            end
            
        end
        
        function visualizeResults(obj, audio, fs, speakers)
            
            % Reverse the process used to attribute speakers to windows to
            % get the indices of the audio clip spoken by the speakers
            speech_vector = preProcessAudio(obj, audio, fs);
            windows = makeAudioWindows(obj, speech_vector, fs, 1, 0.5);
            speaker_indices = getSpeakerIndices(obj, fs, windows, 1, 0.5, speakers);
        end
    end
end