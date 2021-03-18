% TODO:
%   1) Fix bug with probability matrix not lining up with speaker names
%   order
%   2) Some speech not being calculated (when it is determined to be speech
%   by detectSpeech()


classdef (ConstructOnLoad) SpeechProcessing < handle
    
    % Private properties
    properties (SetAccess = private)
        bandpass_filter
        d_vectors = struct();
        d_vectors_total = 0;
        trained_speakers = 0;
        embedding_length = 128;
        window_size = 1; % seconds
        window_overlap = 0; % seconds
        speaker_classifier;
        speaker_names = {};
    end
    
    % Private methods
    methods (Access = protected)
        
        % Calculate the cosine similarity between two vectors
        function cos_sim = vectorSimilarity(~, v1, v2)
            
            cos_sim = dot(v1, v2) / (norm(v1) * norm(v2));
            
        end
        
        % Apply the audio pre-processing to the specified entry
        function [speech_vector, speech_indices] = preProcessAudio(obj, audio, fs)
            
            % Apply bandpass filter
            data = obj.bandpass_filter.filter(audio);
            
            % Normalize audio
            data = data ./ norm(data);
            
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
        function windows = makeAudioWindows(obj, audio, fs)
            
            % Convert window size and overlamp to sample domain
            window_size_samples = obj.window_size * fs;
            window_overlap_samples = obj.window_overlap * fs;
            
            % Pre-allocate the windows matrix
            window_delta_samples = window_size_samples - window_overlap_samples;
            window_amount = ceil(length(audio) / window_delta_samples);
            
            % Check to make sure the clip is long enough to break into
            % windows. If not, return empty array.
            if (window_amount == 0)
                windows = [];
                return;
            end
            
            windows = zeros(window_amount, window_size_samples);
            
            % Check to see if there is enough audio for a single window. If
            % not, pad the end with zeroes
            if (length(audio) < window_size_samples)
                windows(1, 1 : length(audio)) = audio(1 : end);
            % If not, assign the values of the windows matrix
            else
                windows(1, :) = audio(1 : window_size_samples); 
            end
            for row = 2 : window_amount
                % Check to see if the window is too large
                if ((window_size_samples + (window_delta_samples * (row - 1)) - 1) > length(audio))
                    windows(row, 1 : length(audio(window_delta_samples * (row - 1) : end))) = audio(window_delta_samples * (row - 1) : end);
                    continue;
                end
                windows(row, :) = audio(window_delta_samples * (row - 1) : window_size_samples + (window_delta_samples * (row - 1)) - 1);
            end
            
        end
        
        function embeddings = getEmbeddings(obj, audio, fs)
            % Grab audio windows that will be processed
            windows = makeAudioWindows(obj, audio, fs);
            
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
        
        % Returns the of the processed speech vector for the data
        % attributed to each speaker
        function speaker_indices = getSpeechSpeakerIndices(obj, fs, windows, speakers)
                        
            % Convert window size and overlamp to sample domain
            window_size_samples = obj.window_size * fs;
            window_overlap_samples = obj.window_overlap * fs;
            window_delta_samples = window_size_samples - window_overlap_samples;
            
            % Find the indices attributed to each window
            window_indices = zeros(length(windows(:, 1)), 2);
            window_indices(1, :) = [1, length(windows(1, :))];
            for w = 2 : length(windows(:, 1))
                window_indices(w, :) = [window_delta_samples * (w - 1) + 1, window_overlap_samples + (window_delta_samples * w)];
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
            
            % Attribute the last of the data to the current speaker
            speaker_indices(speaker_changes).idx = [start_idx, window_indices(end, 2)];
            speaker_indices(speaker_changes).speaker = current_speaker;
        end
        
        function original_speaker_indices = getOriginalSpeakerIndices(~, speech_indices, speech_speaker_indices)
            
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
                    % If all the speakers have been calculated, attribute
                    % the rest of the points to the speaker and exit the
                    % function
                    if (speaker_change > length(speech_speaker_indices))
%                         for w = i : length(speech_indices(:, 1))
%                             original_speaker_indices(
%                         end
                        return;
                    end
                    start_idx = start_idx + leftover_samples + 1;
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
                    start_idx = start_idx + (speech_speaker_indices(speaker_change - 1).idx(2) - speech_speaker_indices(speaker_change - 1).idx(1)) + 1;
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
        
        % Returns the time that a point was sampled at when given the index
        % and sample rate
        function timeIndices = getTimeForIdx(~, idx, fs)
            
            timeIndices = zeros(1, length(idx));
            for i = 1 : length(idx)
                timeIndices(i) = idx(i) / fs;
            end
        end
        
        % Checks for overlapping speakers and corrects the labels
        function new_labels = checkOverlapping(~, labels)
            
            % Check for transistions in speakers
            current_speaker = string(labels(1));
            for s = 2 : length(labels)
                if (~strcmp(current_speaker, string(labels(s))))
                    % If the transition happens because of an unknown
                    % speaker, ignore it for now
                    if (strcmp("Unknown", string(labels(s))))
                        continue;
                    end
                    current_speaker = string(labels(s));
                    labels(s - 2) = cellstr("Transition Point Within Bounds");
                    labels(s - 1) = cellstr("Transition Point Within Bounds");
                    labels(s) = cellstr("Transition Point Within Bounds");
                end
            end
            
            % Return new array
            new_labels = labels;
        end
    end
    
    % Public methods
    methods (Access = public)
        
        % Constructor requires pre-trained bandpass filter. Need to fix
        % this so it is included when the object is created.
        function obj = SpeechProcessing()
            
            % Check to see if speech_filter can be loaded
            try
                obj.bandpass_filter = load('speech_filter.mat').speech_filter;
            catch
                assert(0, 'Can not find speech_filter.mat. Is it in this directory?');
            end
            
            % Check to see if the path to vggish exists
            try
                addpath('./vggish');
            catch
                assert(0, 'Can not find path to vggish. Is it in this directory?')
            end
        end
        
        function memorizeSpeaker(obj, speaker_audio, fs, name)
                        
            % Check if name already exists
            name_exists = 1;
            for names = 1 : length(obj.speaker_names)
                if (strcmp(char(obj.speaker_names(names)), name))
                    name_exists = 0;
                    break;
                end
            end
            assert(name_exists, 'User already exists in the database.')
            
            % If name doesn't exist, add it to the database
            obj.speaker_names{length(obj.speaker_names) + 1} = name;
            
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
            
            % Store embeddings
            for e = 1 : length(speaker_embeddings)
                for r = 1 : length(speaker_embeddings(e).emb(:, 1))
                    obj.d_vectors(obj.d_vectors_total + 1).emb = speaker_embeddings(e).emb(r, :);
                    obj.d_vectors(obj.d_vectors_total + 1).name = name;
                    obj.d_vectors_total = obj.d_vectors_total + 1;
                end
            end
            
            % Increment the number of speakers remembered
            obj.trained_speakers = obj.trained_speakers + 1;
            
            % Create inputs for fitcknn
            num_obvs = obj.d_vectors_total;
            num_vars = length(speaker_embeddings(1).emb(1, :));
            observations = zeros(num_obvs, num_vars);
            speakers = {};
            for e = 1 : obj.d_vectors_total
                observations(e, :) = obj.d_vectors(e).emb;
                speakers{e} = obj.d_vectors(e).name;
            end
            speakers = speakers.';
            
            % Fit KNN to data points
            obj.speaker_classifier = fitcknn(observations, speakers,...
                'OptimizeHyperparameters', 'all', 'HyperparameterOptimizationOptions',...
                struct('UseParallel', 1));

        end
        
        % Diarize the given audio clip
        function [original_speaker_indices, return_similarities, speaker_names] = diarizeAudioClip(obj, audio, fs, threshold)
            
            % Make sure the model has been trained on at least one speaker
            assert(~isempty(obj.speaker_classifier), 'No speakers have been trained on this model.')
            
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
            % Input variable
            num_obvs = length(speaker_embeddings(:, 1));
            num_vars = length(speaker_embeddings(1, :));
            observations = zeros(num_obvs, num_vars);
            for e = 1 : length(speaker_embeddings(:, 1))
                observations(e, :) = speaker_embeddings(e, :);
            end
            % Use KNN to classify embeddings
            [labels, score, ~] = predict(obj.speaker_classifier, observations);
            
            % Apply threshold to the estimated speakers to cutoff any below
            % the requirement
            for s = 1 : length(score(:, 1))
                % Check if the max of the row is less than the threshold.
                % If so, change speaker to 'Unknown'.
                if (threshold > max(score(s, :)))
                    labels(s) = cellstr('Unknown');
                end
            end
            
            % Check for overlapping in windows
            labels = checkOverlapping(obj, labels);
            
            % Check for multiple speakers in overlapping sections
            %multiple_speaker_idx = checkForMultiSpeakers(audio_window, fs);
                        
            % Get the indices of the original audio clip that each speaker
            % is attributed to
            windows = makeAudioWindows(obj, processed_audio, fs);
            speech_speaker_indices = getSpeechSpeakerIndices(obj, fs, windows, labels);
            original_speaker_indices = getOriginalSpeakerIndices(obj, speech_indices, speech_speaker_indices);
            
            % Return similarities if requested
            if (nargout > 1)
                return_similarities = score;
            end
            if (nargout > 2)
                speaker_names = obj.speaker_names;
            end
            
        end
        
        function unique_speakers = determineUniqueSpeakers(~, speakers)
                        
            % Append the first name
            unique_speakers = speakers(1).speaker;
            
            % Loop through all the speakers
            for s = 2 : length(speakers)
                % Flag to determine unique speaker
                new_speaker = true;
                % Loop through all the names already recorded
                for u = 1 : length(unique_speakers)
                    % If the name is already in the list, ignore it 
                    if (strcmp(unique_speakers(u), speakers(s).speaker))
                        new_speaker = false;
                    end
                end
                
                % If the name is not in the list, append it to the list
                if (new_speaker)
                    unique_speakers = [unique_speakers; speakers(s).speaker];
                end
            end
            
        end
        
        function visualizeResults(obj, audio, fs, speakers)
            
            % Create time vector for plotting
            time = linspace(0, length(audio) / fs, length(audio));
            
            y_max = max(audio);
            y_min = min(audio);
            
            % Plot audio signal over time
            plot(time, audio, 'k');
            xlabel('Time (s)');
            ylabel('Amplitude (?)');
            title('Diarization of Audio Signal');
            hold on;
            
            % Determine number of unique speakers and assign each a color
            unique_speakers = determineUniqueSpeakers(obj, speakers);
            
            % Create color array
            colors = ["red"; "green"; "blue"; "yellow"; "magenta"; "cyan"; "black"; "white"];
            speaker_colors = [];
            for c = 1 : length(unique_speakers)
                speaker_colors = [speaker_colors, colors(c)];
            end
            
            % Loop through the indices and plot them
            for u = 1 : length(speakers)
                
                % Determine the speaker color
                for c = 1 : length(unique_speakers)
                    % Compare current box with unique speaker to find the
                    % correct color
                    if (strcmp(speakers(u).speaker, string(unique_speakers(c))))
                        color = speaker_colors(c);
                        break;
                    end
                end
                
                for i = 1 : length(speakers(u).idx(:, 1))
                    timeIndices = getTimeForIdx(obj, [speakers(u).idx(i,1), speakers(u).idx(i,2)], fs);
                    x = [timeIndices(1), timeIndices(2), timeIndices(2), timeIndices(1)];
                    y = [y_min, y_min, y_max, y_max];
                    
                    patch(x, y, color, 'FaceAlpha', 0.3);
                end
            end
            
            % Apply legend for colored regions
            ls = [];
            for c = 1 : length(unique_speakers)
                eval(['l', num2str(c), ' = plot([NaN,NaN], speaker_colors(', num2str(c), '));']);
                eval(['ls = [ls, l', num2str(c), '];']);
            end
            legend(ls, unique_speakers)
        end
    end
end