classdef tolerantStop < stoppingRule
    %SHARPSTOP Returns stop == 1 as soon as performance decreases
    %   Detailed explanation goes here
    
    properties
        threshold
    end
    
    methods
        function obj = tolerantStop(mode , threshold)
            init( obj , mode , threshold)
        end
        
        function init( obj , mode , threshold)
            
            if threshold > 1 || threshold <0
                error('threshold should be between 0 and 1');
            else
                obj.threshold = threshold;
            end
            
            if mode ~=  1 && mode ~=  2
                error('mode should be set to 1: Increasing or 2: Decreasing')
            else
                obj.mode = mode;
            end
            
            if obj.mode == 1
                obj.previousPerf = 0;
            else
                obj.previousPerf = inf;
            end
        end
        
        % Evaluates a new performance and returns "stop == 1" if perf
        % decreases
        function stop = evaluate(obj, perf)
            
            if obj.mode == 1
                if perf >= obj.previousPerf - obj.threshold
                    stop = 0;
                else
                    stop  = 1;
                end
            else
                if perf <= obj.previousPerf + obj.threshold
                    stop = 0;
                else
                    stop  = 1;
                end
            end
            
            obj.previousPerf = perf;
        end
    end
end