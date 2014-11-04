
classdef Sido < dataset
   
   properties
        %outputFormat
   end
   
   methods
        function obj = Sido(nTr , nTe)
            
            warning('Unbalanced training set!');
            warning('Test labels not available!');
            
            Xtr = load('sido0_train.mat');
            Xtr = full(Xtr.X);
            Xte = load('sido0_test.mat');
            Xte = full(Xte.X);
            Ytr = dlmread('sido0_train.targets');
            
            obj.Y = Ytr;    % NO Yte available!
            
            
            obj.X = [Xtr ; Xte];

            %obj.n = size(Xtr , 1) + size(Xte , 1);
            obj.n = size(Xtr , 1);
            
            obj.d = size(Xtr , 2);
            obj.t = size(Ytr , 2);
                
            if nargin == 0
                
%                 obj.nTr = size(Xtr , 1);
%                 obj.nTe = size(Xte , 1);

                obj.nTr = 10000;
                obj.nTe = 2678;
                
                obj.trainIdx = 1:obj.nTr;
                obj.shuffledTrainIdx = obj.trainIdx;
                obj.testIdx = obj.nTr + 1 : obj.nTr + obj.nTe;
                
            elseif (nargin > 1) && (nTr > 1) && (nTe > 0)
                
                if (nTr > 10000) || (nTe > 2678)
                    error('(nTr > 10000) || (nTe > 2678)');
                end
                
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                obj.trainIdx = 1:obj.nTr;
                obj.shuffledTrainIdx = obj.trainIdx;
                obj.testIdx = 10001:10000 + obj.nTe;
            end
            
            % Reformat output columns
%             if (nargin > 2) && (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
%                 obj.outputFormat = outputFormat;
%             else
%                 display('Wrong or missing output format, set to plusMinusOne by default');
%                 obj.outputFormat = 'plusMinusOne';
%             end
            
%             if strcmp(obj.outputFormat, 'zeroOne')
%                 obj.Y = zeros(obj.n,obj.t);
%             elseif strcmp(obj.outputFormat, 'plusMinusOne')
%                 obj.Y = -1 * ones(obj.n,obj.t);
%             elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
%                 obj.Y = -1/(obj.t - 1) * ones(obj.n,obj.t);
%             end
%                
%             for i = 1:obj.n
%                 obj.Y(i , data.gnd(i)) = 1;
%             end
            
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
            
%             if strcmp(obj.outputFormat, 'zeroOne')
%                 Ypred = zeros(size(Yscores));
%             elseif strcmp(obj.outputFormat, 'plusMinusOne')
%                 Ypred = -1 * ones(size(Yscores));
%             elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
%                 Ypred = -1/(obj.t - 1) * ones(size(Yscores));
%             end

            warning('The dataset is unbalanced, proper classification methods have to be implemented');
            Ypred = sign(Yscores);

        end
            
        % Compute performance measure on the given outputs according to the
        % dataset-specific ranking standard measure
        function perf = performanceMeasure(obj , Y , Ypred)
            
            % Check if Ypred is real-valued. If yes, convert it.
            if obj.hasRealValues(Ypred)
                Ypred = obj.scoresToClasses(Ypred);
            end
            
            % Compute error rate
            numCorrect = 0;
            
            for i = 1:size(Y,1)
                if Y(i) == Ypred(i)
                    numCorrect = numCorrect +1;
                end
            end
            
            perf = 1 - (numCorrect / size(Y,1));
        end
        
        % Compute random permutation of the training set indexes
        function shuffleTrainIdx(obj)
            obj.shuffledTrainIdx = randperm(obj.nTr);
        end
        
        function getTrainSet(obj)
            
        end
        
        function getTestSet(obj)
            
        end
        
   end % methods
end % classdef