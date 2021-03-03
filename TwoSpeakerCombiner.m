function [audio, sampleRate] = TwoSpeakerCombiner(file1, file2);
    
    % Read audio clips
    [audio1, fs1] = audioread(file1);
    [audio2, fs2] = audioread(file2);
    
    % Combine audio clips
    audio = [audio1.', audio2.'];
    
    % Determine which sample rate to return
    if (fs1 == fs2)
        sampleRate = fs1;
    end
    
    if (fs1 < fs2)
        sampleRate = fs1;
    end
    
    if (fs2 < fs1)
        sampleRate = fs2;
    end
    
end