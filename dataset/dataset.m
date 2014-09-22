
classdef dataset < handle
   
   % Define an event
   properties
        n
        d
        t
        X
        Y
   end
   
   methods
        function obj = dataset(fname)
            data = load(fname);
            obj.X = data.X;
            obj.Y = data.Y;
            obj.n = size(data.X , 1);
            obj.d = size(data.X , 2);
            obj.t = size(data.y , 2);
        end
        function perf = performanceMeasure(obj , Y , Yhat)
            
        end
   end % methods
end % classdef