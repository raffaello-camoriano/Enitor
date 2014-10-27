
classdef USPS < dataset
   
   properties
   
   end
   
   methods
        function obj = USPS
            data = load('USPS.mat');
            
            obj.X = data.fea;
            
            obj.n = size(obj.X , 1);
            obj.nTr = 7291;
            obj.nTe = 2007;
            obj.d = size(obj.X , 2);
            obj.t = max(data.gnd);
            
            % reformat output columns
            obj.Y = zeros(obj.n,obj.t);
            
            for i = 1:obj.n
                
                obj.Y(i , data.gnd(i)) = 1;
            
            end
            
            obj.trainIdx = 1:obj.nTr;
            obj.shuffledTrainIdx = obj.trainIdx;
            obj.testIdx = obj.nTr+1:obj.nTr+obj.nTe;
            
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
        
        % Compute performance measure on the given outputs according to the
        % USPS dataset-specific ranking standard measure
        function perf = performanceMeasure(obj , Y , Ypred)
            
            % Error rate

            numCorrect = 0;
            
            for i = 1:size(Y,1)
                if sum(Y(i,:) == Ypred(i,:)) == size(Y,2)
                    numCorrect = numCorrect +1;
                end
            end
            
            perf = 1 - (numCorrect / size(Y,1));
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