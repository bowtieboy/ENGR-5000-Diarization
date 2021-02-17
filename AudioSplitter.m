function [windows,samplesPerWindow] = AudioSplitter(audio, sampleFreq, timeWindowLength, filterOrder)
% AUDIOSPLITTER Splits vector of floats into chunks of time
% depending on the sample frequency of the audio
%   WINDOWS = AUDIOSPLITTER(AUDIO, SAMPLEFREQ, WINDOWSIZE) returns matrix
%   where the rows are the time slices and the columns are samples in each
%   time slice.

    samplesPerWindow = (sampleFreq * timeWindowLength) + filterOrder;

    % Grab the initial values
    samples = length(audio);
    
    windowAmount = floor(samples / samplesPerWindow);
    
    % Pre-allocate the return matrix
    windows = zeros(windowAmount, samplesPerWindow);
    
    % Loop through the audio vector and assign the values
    windows(1, :) = audio(1 : samplesPerWindow);
    for row = 2 : windowAmount
        for col = 1 : samplesPerWindow
            if (col <= filterOrder)
                windows(row, col) = windows(row - 1, (samplesPerWindow - filterOrder) + col);
                continue;
            end
            windows(row, col) = audio(((row - 1) * samplesPerWindow) + col - filterOrder);
        end
    end
end

