
classdef dataset < handle
   
   properties
        n
        nTr
        nTe
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
            if  nargin > 0            
                data = load(fname);
                obj.X = data.X;
                obj.Y = data.Y;
                obj.n = size(data.X , 1);
                obj.d = size(data.X , 2);
                obj.t = size(data.y , 2);
            end
        end       
        
        function perf = performanceMeasure(obj , Y , Ypred )
            % Error Rate is computed
            
            
            
        end
   end % methods
end % classdef