
classdef dataset < handle
   
   properties
       
        problemType     % regression or classification
        
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
                
                % Set problem type

                obj.problemType = 'classification';
                for i = 1:size(obj.Y,1)
                    for j = 1:size(obj.Y,2)

                        if mod(obj.Y(i,j),1) ~= 0
                            obj.problemType = 'regression';
                        end
                    end
                end
            end
        end       
        
        function perf = performanceMeasure(obj , Y , Ypred )
            % Error Rate is computed
            
            
            
        end
   end % methods
end % classdef