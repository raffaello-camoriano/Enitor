classdef stoppingRule < handle
    %STOPPINGRULE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        previousPerf    % Performance value at the previous step
        mode            % 1: Increasing ; 2: Decreasing
    end
    
    methods (Abstract)
%         stoppingRule(mode)
        init( obj , mode)
        evaluate(perf)
        reset()
    end
    
end

