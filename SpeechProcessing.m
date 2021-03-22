% TODO:
%   Use Google's diarization to refine speaker identification points and
%   words

classdef (ConstructOnLoad) SpeechProcessing < handle
    
    % Private properties
    properties (SetAccess = private)
        % The preprocessing speech filter
        bandpass_filter
        % All fitted d_vectors from training
        d_vectors = struct();
        % Count of all d_vectors
        d_vectors_total = 0;
        % Count of all trained speakers
        trained_speakers = 0;
        % How long the audio embedding is
        embedding_length = 128;
        % Length of audio window
        window_size = 1; % seconds
        % Length of audio overlap
        window_overlap = 0; % seconds
        % kNN classifier model
        speaker_classifier;
        % List of all trained speakers
        speaker_names = {};
        % Google speech-to-text API
        speechObject;
        % Expands window for checking who said which word
        time_shift = 0.2;
    end
    
    % Private methods
    methods (Access = protected)
        
        % Calculate the cosine similarity between two vectors
        function cos_sim = vectorSimilarity(~, v1, v2)
            
            cos_sim = dot(v1, v2) / (norm(v1) * norm(v2));
            
        end
        
        % Apply the audio pre-processing to the specified entry
        function [speech_vector, new_fs, speech_indices, norm_audio] = preProcessAudio(obj, audio, fs)
            
            % Make sure audio is sampled at the correct frequency, and if
            % not resample it
            if (fs > 16000)
                audio = resampleAudio(obj, audio, fs, 16000);
                fs = 16000;
            end
            
            % Normalize audio
            audio = (audio - min(audio)) ./ (max(audio) - min(audio));
                        
            % Apply bandpass filter
            data = obj.bandpass_filter.filter(audio);
            
            % Cut out silences
            speechIdx = detectSpeech(data.', fs);
            speech_vector = [];
            for i = 1 : length(speechIdx(:, 1))
                speech_vector = [speech_vector, data(speechIdx(i, 1) : speechIdx(i, 2))];
            end
            
            if (nargout > 1)
                new_fs = fs;
            end
            
            if (nargout > 2)
                speech_indices = speechIdx;
            end
            
            if (nargout > 3)
                norm_audio = data;
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
        
        % Returns the index that a point was sampled at when given the time
        % and sample rate
        function vectorIndices = getIdxForTime(~, time, fs)
            vectorIndices = zeros(1, length(time));
            for i = 1 : length(time)
                vectorIndices(i) = time(i) * fs;
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
                    if (s > 2)
                        labels(s - 2) = cellstr("Transition Point Within Bounds");
                    end
                    labels(s - 1) = cellstr("Transition Point Within Bounds");
                    labels(s) = cellstr("Transition Point Within Bounds");
                end
            end
            
            % Return new array
            new_labels = labels;
        end
        
        % Re-sample the audio vector
        function resampled_audio = resampleAudio(~, audio, fs, desired_rate)
            
            % Ensure the desired rate is lower than the original
            assert(fs > desired_rate, 'New sample rate must be lower than the original.');
            
            % Calculate numerator and denominator to achieve desired sample
            % rate
            ratio = desired_rate / fs;
            [num, denom] = rat(ratio);
            
            % Resample audio
            resampled_audio = resample(audio, num, denom);
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
            
            % Check to see if the path to speech2text exists
            try
                addpath('./speech2text');
                obj.speechObject = speechClient('Google','languageCode','en-US',...
                    'sampleRateHertz',16000,'enableWordTimeOffsets',true,...
                    'enableSpeakerDiarization',true);
            catch
                assert(0, 'Can not find path to speech2text. Is it in this directory?')
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
                [processed_audio(clips).audio, fs] = preProcessAudio(obj, speaker_audio(clips).audio, fs);
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
        function [original_speaker_indices, return_similarities, speaker_names, preprocessed_audio] = diarizeAudioClip(obj, audio, fs, threshold)
            
            % Make sure the model has been trained on at least one speaker
            assert(~isempty(obj.speaker_classifier), 'No speakers have been trained on this model.')
            
            % Make sure audio clip is long enough, otherwise error out
            assert((length(audio) / fs) >= 1, 'Audio clip is not long enough to diarize.');
            
            % Pre-process audio
            disp('Pre-processing audio clip.');
            [processed_audio, fs, speech_indices] = preProcessAudio(obj, audio, fs);
            
            % Calculate speaker embeddings
            disp('Extracting d-vectors from audio clip.');
            speaker_embeddings = getEmbeddings(obj, processed_audio, fs);
            
            
            % Use KNN to classify embeddings
            disp('Determining the similarities between the d-vectors and recorded speakers.');
            [labels, score, ~] = predict(obj.speaker_classifier, speaker_embeddings);
            
            % Apply threshold to the estimated speakers to cutoff any below
            % the requirement
            for s = 1 : length(score(:, 1))
                % Check if the max of the row is less than the threshold.
                % If so, change speaker to 'Unknown'.
                if (threshold > max(score(s, :)))
                    labels(s) = cellstr('Unknown');
                end
            end
            
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
            if (nargout > 3)
                preprocessed_audio = processed_audio;
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
        
        function visualizeResults(obj, audio, fs, speakers, uiax)
            
            % Re-sample the audio to line it with the model
            if (fs > 16000)
                audio = resampleAudio(obj, audio, fs, 16000);
                fs = 16000;
            end
            
            % Create time vector for plotting
            time = linspace(0, length(audio) / fs, length(audio));
            
            % Define min and max for squares
            y_max = max(audio);
            y_min = min(audio);
            
            % Create new axes if none is given
            fig = uifigure();
            if (nargin > 4)
                ax = uiax;
            else
                ax = uiaxes(fig);
            end
            
            
            % Plot audio signal over time
            plot(ax, time, audio, 'k');
            xlabel('Time (s)');
            ylabel('Amplitude');
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
                
                for i = 1 : length(speakers(u).times(:, 1))
                    time_indices = speakers(u).times(i, :);
                    x = [time_indices(1), time_indices(2), time_indices(2), time_indices(1)];
                    y = [y_min, y_min, y_max, y_max];
                    
                    patch(ax, x, y, color, 'FaceAlpha', 0.3);
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
        
        function [true_speakers] = annotateAudio(obj, audio, fs, threshold)
            
            % Diarize the audio segment to determine who spoke when
            [original_speaker_indices, ~, ~] = diarizeAudioClip(obj, audio, fs, threshold);
            
            % Pre-process audio
            [~, new_fs, ~, audio] = preProcessAudio(obj, audio, fs);
            fs = new_fs;
            
            % Perform speech-to-text on the audio
            disp('Performing Speech-to-text cloud computing.');
            table_out = speech2text(obj.speechObject,audio,fs);
            
            % Determine number of unique speakers and assign each a color
            unique_speakers = determineUniqueSpeakers(obj, original_speaker_indices);
            
            % Convert indices to time stamps
            original_speaker_times = struct();
            for s = 1 : length(original_speaker_indices)
                % Initialize variables associated with each speaker
                original_speaker_times(s).speaker = original_speaker_indices(s).speaker;
                original_speaker_times(s).times = [];
                original_speaker_times(s).word_times = [];
                for i = 1 : length(original_speaker_indices(s).idx(:, 1))
                    time_idx = original_speaker_indices(s).idx(i, :);
                    speaker_times = getTimeForIdx(obj, [time_idx(1), time_idx(2)], fs);
                    original_speaker_times(s).times = [original_speaker_times(s).times; speaker_times];
                end
            end
            
            % Combine entries of speakers that occured more than once
            concat_speaker_times = struct();
            for unique = 1 : length(unique_speakers)
                times = [];
                for speaker = 1 : length(original_speaker_times)
                    if (strcmp(unique_speakers(unique), original_speaker_times(speaker).speaker))
                        times = [times; original_speaker_times(speaker).times];
                    end
                end
                concat_speaker_times(unique).speaker = string(unique_speakers(unique));
                concat_speaker_times(unique).times = times;
                concat_speaker_times(unique).word_times = [];
                concat_speaker_times(unique).idx = [];
            end
            
            % Set previous speaker variable to zero for initialization
            previous_speaker = 0;
            % Loop through the tables of speakers identified by Google
            for table = 1 : height(table_out)
                time_stamps = table_out.TimeStamps{table};
                % Reset all speaker points 
                speaker_points = zeros(height(time_stamps), length(unique_speakers));
                % Empty word list
                word_list = string();
                % Empty word timings list
                word_timings = zeros(length(time_stamps.startTime), 2);
                % Loop through word timings to check who said each word
                for w = 1 : length(time_stamps.startTime)
                    % Get the time window for the word
                    word_start = time_stamps.startTime(w);
                    word_end = time_stamps.endTime(w);
                    % Convert from cell to string
                    word_start = word_start{1};
                    word_end = word_end{1};
                    % Remove unit from string
                    word_start = erase(word_start, 's');
                    word_end = erase(word_end, 's');
                    % Convert from string to double
                    word_start = str2double(word_start);
                    word_end = str2double(word_end);
                    % Store word timings as vector
                    current_word_timings = [word_start, word_end];
                    % Append timings to overall list
                    word_timings(w, :) = current_word_timings;
                    % Average time to find middle of word
                    word_time = ((word_start + word_end) / 2);
                    % Grab the current word
                    current_word = time_stamps.word(w);
                    current_word = current_word{1};
                    % If first word in list, replace empty entry
                    if (w == 1)
                        word_list = string(current_word);
                    else
                        % Append current word to list
                        word_list = [word_list; current_word];
                    end
                    word_attributed = false;

                    % Determine who said the word by comparing the time the
                    % word was said to who was speaking at that time
                    for s = 1 : length(concat_speaker_times)
                        % Set the column for the speaker point to be
                        % attriubted
                        for c = 1 : length(unique_speakers)
                            if (strcmp(concat_speaker_times(s).speaker, unique_speakers(c)))
                                speaker_col = c;
                            end
                        end
                        for t = 1 : length(concat_speaker_times(s).times(:, 1))
                            % Determine the start and end time for the current
                            % speech segment
                            speaker_start = concat_speaker_times(s).times(t, 1);
                            speaker_end = concat_speaker_times(s).times(t, 2);
                            % If the words fall into this segment, attribute
                            % the word to the speaker
                            if ((speaker_start <= (word_time + obj.time_shift)) && (speaker_end >= (word_time - obj.time_shift)))
                                speaker_points(w, speaker_col) = 1;
                                word_attributed = true;
                                break;
                            end
                        end

                        % If the speaker was already found, stop searching
                        if (word_attributed)
                            break;
                        end
                    end
                end
                % Use Google's diarization to assume seperation point
                % for words and add words to the speakers
                assumed_speaker = 1;
                current_points = 0;
                % Find who had the most words attirubted to them
                for c = 1 : length(speaker_points(1, :))
                    % Assume that the previous speaker can not be the same
                    % as the current, otherwise the audio would not have
                    % been diarized
                    if (c == previous_speaker)
                        continue;
                    end
                    % If it is not the same speaker, check to see if the
                    % current speaker had more words attributed to them
                    % than the previous
                    if (sum(speaker_points(:, c)) > current_points)
                        current_points = sum(speaker_points(:, c));
                        assumed_speaker = c;
                        previous_speaker = assumed_speaker;
                    end
                end
                % Grab the name of the current speaker
                concat_speaker_times(assumed_speaker).words = word_list;
                concat_speaker_times(assumed_speaker).word_times = word_timings;
            end
            
            % Remove any speakers that did not speak, and record the times
            % when they spoke
            speakerless_times = [];
            true_speakers = concat_speaker_times;
            deleted_speakers = 0;
            for speaker = 1 : length(concat_speaker_times)
                if(isempty(concat_speaker_times(speaker).words))
                    % If person did not speak, record their times and
                    % remove them from the list
                    speakerless_times = [speakerless_times; concat_speaker_times(speaker).times];
                    true_speakers(speaker - deleted_speakers) = [];
                    deleted_speakers = deleted_speakers + 1;
                end
            end
            
            % Attribute the speakerless windows the the speaker with the
            % closest windows
            if(~isempty(speakerless_times))
                for t = 1 : length(speakerless_times(:, 1))
                    current_sl_w = speakerless_times(t, :);
                    min_distance = inf;
                    for speaker = 1 : length(true_speakers)
                        for w = 1 : length(true_speakers(speaker).times(:, 1))
                            current_w = true_speakers(speaker).times(w, :);
                            current_distance = abs(current_w(1) - current_sl_w(1)) + abs(current_w(2) - current_sl_w(2));

                            % If the current distance is the smallest, record
                            % the distance and the speaker
                            if (current_distance < min_distance)
                                min_distance = current_distance;
                                most_likely_speaker = speaker;
                            end
                        end
                    end

                    % Append the window to the times of the most likely speaker
                    true_speakers(most_likely_speaker).times = [true_speakers(most_likely_speaker).times; current_sl_w];
                end
            end
            
            % Sort the time list for each speaker
            for speaker = 1 : length(true_speakers)
                true_speakers(speaker).times = sort(true_speakers(speaker).times);
            end
            
            % Modify speaker timings based on the diarization transition
            % point
            for speaker = 1 : length(true_speakers)
                % Change speaker start to be at the end of the last speaker
                start_time = true_speakers(speaker).word_times(1);
                % Change speaker ending to be at the end of their last word
                end_time = true_speakers(speaker).word_times(end);
                % Start the index at 1
                time_idx = 1;
                while (time_idx < length(true_speakers).times(:, 1))
                    % Store the window being used for the calculation
                    current_window = true_speakers(speaker).times(time_idx, :);
                    
                    % If the first word said occurs before the speaker
                    % started speaking, attirbute it to the previous
                    % speaker
                    if (current_window(1) < start_time)
                        % If this is the first speaker, delete the entry
                        if (speaker == 1)
                            true_speakers(speaker).times(time_idx, :) = [];
                            continue;
                        end
                        
                        % If this is not the first speaker, attribute this
                        % window to the previous speaker
                        true_speakers(speaker - 1).times = [true_speakers(speaker - 1).times; current_window];
                        true_speakers(speaker).times(time_idx, :) = [];
                        continue;
                    end
                    
                    % If last word said occurs before the speaker section
                    % ends, end the speaker section at the end of their
                    % last word spoken
                    if (current_window(2) > end_time)
                        
                        % If there is a speaker after them, assign the
                        % speach window to them
                        if (~(speaker == length(true_speakers)))
                            true_speakers(speaker + 1).times = [current_window; true_speakers(speaker + 1).times];
                        end
                        
                        true_speakers(speaker).times(time_idx, :) = [];
                        continue;
                    end
                    
                    % Iterate time_idx
                    time_idx = time_idx + 1;
                end
                
                % Convert the times back to indices for plotting function
                for i = 1 : length(true_speakers(speaker).times(:, 1))
                    time_to_convert = true_speakers(speaker).times(i, :);
                    true_speakers(speaker).idx = [true_speakers(speaker).idx; getIdxForTime(obj, time_to_convert, fs)];
                end
            end            
        end
    end
end