classdef horizonSharpStop < stoppingRule
    %horizonSharpStop  Returns stop == 1 as soon as performance decreases
    %for more than "horizon" iterations
    %   Detailed explanation goes here
    
    properties
        horizonLength
        conditionCount
    end
    
    methods
        function obj = horizonSharpStop (mode, horizonLength )
            init( obj , mode , horizonLength)
        end
        
        function init( obj , mode , horizonLength)
            
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
            
            if horizonLength > 0
                obj.horizonLength = horizonLength;
            else
                error('horizonLength must be > 0')
            end
            obj.conditionCount = 0;
        end
        
        % Evaluates a new performance and returns "stop == 1" if perf
        % decreases
        function stop = evaluate(obj, perf)
            
            if obj.mode == 1
                if perf > obj.previousPerf
                    stop = 0;
                    obj.conditionCount = 0;
                    obj.previousPerf = perf;
                else
                    obj.conditionCount = obj.conditionCount + 1;
                    if obj.conditionCount  == obj.horizonLength
                        stop  = 1;
                    else
                        stop = 0;
                    end
                end
            else
                if perf < obj.previousPerf
                    stop = 0;
                    obj.conditionCount = 0;
                    obj.previousPerf = perf;
                else
                    obj.conditionCount = obj.conditionCount + 1;
                    if obj.conditionCount  == obj.horizonLength
                        stop  = 1;
                    else
                        stop = 0;
                    end
                end
            end
        end
        
        function reset(obj)
            if obj.mode == 1
                obj.previousPerf = 0;
            else
                obj.previousPerf = inf;
            end
            obj.conditionCount = 0;
        end
    end
end