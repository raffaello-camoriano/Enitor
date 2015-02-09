classdef nystrom < handle
    %RANDOMFEATURES Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        d
        
        kerPar
        rng
     
        numMapParGuesses
        
        numNysParGuesses    % Number of sampled columns guesses
        
        currentPar
        currentParIdx
        
        X
        
        verbose
    end
end
