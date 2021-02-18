% TODO:
%   1) Fix issue where indices stored but not data

function speechData = SpeechSeperator(audioWindows, sampleFreq)
    
    speechData = struct();
    windowAmount = length(audioWindows(:, 1));
    entry = 1;

    for row = 1 : windowAmount
        speechIdx = detectSpeech(audioWindows(row, :).', sampleFreq);
        for windows = 1 : (length(speechIdx(:, 1)))
            speechData(entry).speech = audioWindows(row, speechIdx(windows, 1) : speechIdx(windows, 2));
            speechData(entry).originalIdx = [speechIdx(windows, 1), speechIdx(windows, 2)];
            speechData(entry).originalRow = row;
            speechData(entry).segmentsFromRow = length(speechIdx(:, 1));
            entry = entry + 1;
        end
    end
end