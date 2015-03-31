classdef sharpStop < stoppingRule
    %SHARPSTOP Returns stop == 1 as soon as performance decreases
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function obj = sharpStop(mode)
            init( obj , mode)
        end
        
        function init( obj , mode)
            
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
        function stop = evaluate(perf)
            
            if obj.mode == 1
                if perf > obj.previousPerf
                    stop = 0;
                else
                    stop  = 1;
                end
            else
                if perf < obj.previousPerf
                    stop = 0;
                else
                    stop  = 1;
                end
            end
        end
    end
end