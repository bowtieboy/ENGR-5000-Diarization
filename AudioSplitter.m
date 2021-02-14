function windows = AudioSplitter(audio, samplesPerWindow)
% AUDIOSPLITTER Splits vector of floats into chunks of time
% depending on the sample frequency of the audio
%   WINDOWS = AUDIOSPLITTER(AUDIO, SAMPLEFREQ, WINDOWSIZE) returns matrix
%   where the rows are the time slices and the columns are samples in each
%   time slice.

    % Grab the initial values
    samples = length(audio);
    windowAmount = floor(samples / samplesPerWindow);
    
    % Pre-allocate the return matrix
    windows = zeros(windowAmount, samplesPerWindow);
    
    % Loop through the audio vector and assign the values
    for row = 1 : windowAmount
        for col = 1 : samplesPerWindow
            windows(row, col) = audio(((row - 1) * samplesPerWindow) + col);
        end
    end
end

