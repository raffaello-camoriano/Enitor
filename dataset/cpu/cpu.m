
classdef cpu < dataset
   
   properties
        outputFormat
        linearity
        noise
        
        minY
        maxY
        Yclasses
   end
   
   methods
        function obj = cpu(nTr , nTe, ~, shuffleTraining, shuffleTest, shuffleAll)

            % Call superclass constructor with arguments
            obj = obj@dataset([], shuffleTraining, shuffleTest, shuffleAll);
            
            % Number of samples
            obj.nTrTot = 6554;
            obj.nTeTot = 1638;
            if nTr > obj.nTrTot || nTe > obj.nTeTot
                error('Too many samples required')
            end
            obj.nTr = nTr;
            obj.nTe = nTe;
            
            data = load('cpu_act.data');
            
            obj.X = data(:,1:end-1);
            obj.Y = data(:,end);
            
            obj.d = size(obj.X,2);
            data = [];
            
            % Scale data
            obj.X = obj.scale(obj.X);  
            
            obj.n = size(obj.X , 1);
            
            % Set t (number of classes)
            obj.t = 1;
            
            % Set training and test indexes
            obj.trainIdx = 1:obj.nTr;
            obj.testIdx = obj.nTrTot + 1 : obj.nTrTot + obj.nTe;
            

            % Shuffling

            obj.shuffleTraining = shuffleTraining;
            if shuffleTraining == 1
                obj.shuffleTrainIdx();
            end
            
            obj.shuffleTest = shuffleTest;
            if shuffleTest == 1
                obj.shuffleTestIdx();
            end
            
            obj.shuffleAll = shuffleAll;
            if shuffleAll == 1
                obj.shuffleAllIdx();
            end            
            
            obj.problemType = 'regression';
        end
            
        % Compute predictions matrix from real-valued scores matrix
%         function Ypred = scoresToClasses(obj , Yscores)    
% 
%             Ypred = round(Yscores*(obj.maxY-obj.minY)+obj.minY);
%         end
%         
        % Compute performance measure on the given outputs
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            
            % RMSE
            perf = sqrt(sum((Y - Ypred).^2)/size(Y,1));
            
%             Ypred = obj.scoresToClasses(Ypred);
% 
%             % Compute error rate
%             numCorrect = 0;
%             for i = 1:size(Y,1)
%                 if obj.Yclasses(i) == Ypred(i,:)
%                     numCorrect = numCorrect +1;
%                 end
%             end
% 
%             perf = 1 - (numCorrect / size(Y,1));       
%             
%                 C = transpose(bsxfun(@eq, Y', Ypred'));
%                 D = sum(C,2);
%                 E = D == size(Y,2);
%                 numCorrect = sum(E);
%                 perf = 1 - (numCorrect / size(Y,1));           
        end
        
        % Scales matrix M between 0 and 1
        function Ms = scale(obj , M)

            mx = max(M);
            mn = min(M);
            
            delta = mx - mn;
            Ms = transpose(bsxfun(@minus, M', mn'));
            Ms = transpose(bsxfun(@rdivide, Ms', delta'));
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
    end % methods
end % classdef