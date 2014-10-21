
classdef USPS < dataset
   
   properties
   
   end
   
   methods
        function obj = USPS
            data = load('USPS.mat');
            
            obj.X = data.fea;
            obj.Y = data.gnd;
            obj.n = size(obj.X , 1);
            obj.nTr = 7291;
            obj.nTe = 2007;
            obj.d = size(obj.X , 2);
            obj.t = max(obj.Y);
            
            obj.trainIdx = 1:obj.nTr;
            obj.shuffledTrainIdx = obj.trainIdx;
            obj.testIdx = obj.nTr+1:obj.nTr+obj.nTe;
        end
        
        % Compute performance measure on the given outputs according to the
        % USPS dataset-specific ranking standard measure
        function perf = performanceMeasure(obj , Y , Ypred)
            % Error rate
            equal = (Y == Ypred);
            numCorrect = sum(equal);
            perf = numCorrect / size(Y,1);
        end
        
        % Compute random permutation of the training set indexes
        function shuffleTrainIdx(obj)
            obj.shuffledTrainIdx = randperm(7291);
        end
        
        function getTrainSet(obj)
            
        end
        
        function getTestSet(obj)
            
        end
        
   end % methods
end % classdef