
classdef USPS < handle
   
   properties
        n
        d
        t
        X
        Y
   end
   
   methods

        function obj = dataset()
            data = load('USPS.mat');
            data = data.data;
            
            obj.X = zeros(size(data,2)*size(data,3),size(data,1));
            for i = 1:size(data , 3)
              obj.X((i-1)*size(data,2)+1:i*size(data,2),:) = data(:,:,i)';
            end
            
            obj.Y = 1:size(data , 3);
            obj.Y = interp1(obj.Y , obj.Y, linspace(0.5, size(data , 3)+0.5, size(obj.X,1)), 'nearest'  , 'extrap');
            obj.n = size(obj.X , 1);
            obj.d = size(obj.X , 2);
            obj.t = size(data , 3);
        end
        function perf = performanceMeasure(obj , Y , Ypred)
            % Error rate
            equal = (Y == Ypred);
            numCorrect = sum(equal);
            perf = numCorrect / size(Y,1);
        end
   end % methods
end % classdef