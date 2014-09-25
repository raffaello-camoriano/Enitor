classdef filter < handle
    %FILTER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        rng
        Yreg
    end
    
    methods ( Abstract )
        %obj = kernel();
        init(obj , K , Y);
        range(obj, numGuesses);
        compute(obj , lambda);
    end
end

