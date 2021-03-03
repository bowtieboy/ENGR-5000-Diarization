function speechData = SpeechSeperator(audioWindows, sampleFreq)
    
    % Final struct that will be returned
    speechData = struct(); 
    windowAmount = length(audioWindows(:, 1));
    entry = 1;

    for row = 1 : windowAmount
        % Grab indices of detected speech
        speechIdx = detectSpeech(audioWindows(row, :).', sampleFreq);
        if (isempty(speechIdx))
            continue;
        end
        for windows = 1 : (length(speechIdx(:, 1)))
            % Store audio of the detected speech
            speechData(entry).speech = audioWindows(row, speechIdx(windows, 1) : speechIdx(windows, 2));
            % Store the index associated with the detected speech
            speechData(entry).originalIdx = [speechIdx(windows, 1), speechIdx(windows, 2)];
            % Store the original row where the data is from
            speechData(entry).originalRow = row;
            % Record how many much seperate speech is in the row
            speechData(entry).segmentsFromRow = length(speechIdx(:, 1));
            entry = entry + 1;
        end
    end
end