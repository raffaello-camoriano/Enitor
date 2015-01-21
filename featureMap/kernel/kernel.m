classdef kernel < handle
    %KERNEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        %rng
        K
        verbose
    end
    
    methods ( Abstract )
        %obj = kernel();
        %init(obj , X , Y);
        range(obj, numGuesses);
        compute(obj , sigma);
    end
end

