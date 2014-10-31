
classdef USPS < dataset
   
   properties
   
   end
   
   methods
        function obj = USPS(nTr , nTe)
            
            data = load('USPS.mat');
            
            obj.X = data.fea;

            obj.n = size(obj.X , 1);
            obj.d = size(obj.X , 2);
            obj.t = max(data.gnd);
            
            % reformat output columns
            obj.Y = zeros(obj.n,obj.t);

            for i = 1:obj.n
                obj.Y(i , data.gnd(i)) = 1;
            end
                
            if nargin == 0
                
                obj.nTr = 7291;
                obj.nTe = 2007;

                obj.trainIdx = 1:obj.nTr;
                obj.shuffledTrainIdx = obj.trainIdx;
                obj.testIdx = obj.nTr+1:obj.nTr+obj.nTe;
                
            elseif (nargin >1) && (nTr > 1) && (nTe > 0)
            
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                if (nTr > 7291) || (nTe > 2007)
                    error('(nTr > 7291) || (nTe > 2007)');
                end
                
                obj.trainIdx = 1:obj.nTr;
                obj.shuffledTrainIdx = obj.trainIdx;
                obj.testIdx = 7292:7291+obj.nTe;
                
            end
            
            % Set problem type
                
            if obj.hasRealValues(obj.Y)
                obj.problemType = 'regression';
            else
                obj.problemType = 'classification';
            end
        end
        
        % Checks if matrix Y contains real values. Useful for
        % discriminating between classification and regression, or between
        % predicted scores and classes
        function res = hasRealValues(obj , M)
        
            res = false;
            for i = 1:size(M,1)
                for j = 1:size(M,2)
                    if mod(M(i,j),1) ~= 0
                        res = true;
                    end
                end
            end
        end
        
        % Compute predictions matrix from real-valued scores matrix
        function Ypred = scoresToClasses(obj , Yscores)    
            
            Ypred = zeros(size(Yscores));
            for i = 1:size(Ypred,1)
                [~,maxIdx] = max(Yscores(i,:));
                Ypred(i,maxIdx) = 1;
            end
        end
            
        % Compute performance measure on the given outputs according to the
        % USPS dataset-specific ranking standard measure
        function perf = performanceMeasure(obj , Y , Ypred)
            
            % Check if Ypred is real-valued. If yes, convert it.
            if obj.hasRealValues(Ypred)
                Ypred = obj.scoresToClasses(Ypred);
            end
            
            % Compute error rate
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