
classdef dataset < handle
   
   properties
        n
        d
        t
        
        X
        Y
        
        trainIdx
        testIdx
        shuffledTrainIdx
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
        
        function perf = performanceMeasure(obj , Y , Ypred  'rm)
            % Error Rate is computed
            
            
            
        end
   end % methods
end % classdef