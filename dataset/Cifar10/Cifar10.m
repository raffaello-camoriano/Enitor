
classdef Cifar10 < dataset
   
   properties
        outputFormat
        classes
   end
   
   methods
        function obj = Cifar10(nTr , nTe, outputFormat , classes)
            
            % Read files
            if ~isempty(nTr)
                numBatches = ceil(nTr/10000);
            else
                numBatches = 5;
            end
            for i  = 1:numBatches
                traindata{i} = load(['data_batch_',num2str(i),'.mat']);
            end
            
            % Load training batches
            gnd = [];
            obj.X = [];
            for i = 1:numBatches
                obj.X = [ obj.X ; traindata{i}.data ];
                gnd = [gnd ; traindata{i}.labels];
            end
            traindata = [];
                
            obj.nTr = size(obj.X,1);
            obj.nTrTot = obj.nTr;
                        
            % Load test batches
            testdata = load('test_batch.mat');
            obj.X = [ obj.X ; testdata.data ];
            gnd = [gnd ; testdata.labels];
            testdata = [];
            
            obj.X = double(obj.X);
%             obj.X = obj.scale(obj.X);
            obj.X = obj.X/255;
            
            obj.n = size(obj.X , 1);
            obj.nTe = obj.n - obj.nTr;
            obj.nTeTot = obj.nTe;
            
            obj.d = size(obj.X , 2);
            
            % Set number of classes
            if ~isempty(classes)
                obj.classes = classes;
                obj.t = numel(classes);
            else
                obj.classes = 1:10;
            end
            obj.t = numel(obj.classes);
            
            if nargin == 0

                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr + 1 : obj.nTr + obj.nTe;
                
            elseif (nargin >1)
                
                if (nTr < 2) || (nTe < 1) ||(nTr > obj.nTrTot) || (nTe > obj.nTeTot)
                    error('(nTr > obj.nTrTot) || (nTe > obj.nTeTot)');
                end
                
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                tmp = randperm( obj.nTrTot);                            
                obj.trainIdx = tmp(1:obj.nTr);          
                
                tmp = obj.nTrTot + randperm( obj.nTeTot );
                obj.testIdx = tmp(1:obj.nTe);
            end
            
            obj.shuffleTrainIdx();
            obj.shuffleTestIdx();
            
            % Reformat output columns
            if (nargin > 2) && (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
                obj.outputFormat = outputFormat;
            else
                display('Wrong or missing output format, set to plusMinusOne by default');
                obj.outputFormat = 'plusMinusOne';
            end
            
            if strcmp(obj.outputFormat, 'zeroOne')
                obj.Y = zeros(obj.n,obj.t);
            elseif strcmp(obj.outputFormat, 'plusMinusOne')
                obj.Y = -1 * ones(obj.n,obj.t);
            elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                obj.Y = -1/(obj.t - 1) * ones(obj.n,obj.t);
            end
            
            for i = 1:obj.n
                obj.Y(i , gnd(i) + 1) = 1;
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
            
            if strcmp(obj.outputFormat, 'zeroOne')
                Ypred = zeros(size(Yscores));
            elseif strcmp(obj.outputFormat, 'plusMinusOne')
                Ypred = -1 * ones(size(Yscores));
            elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                Ypred = -1/(obj.t - 1) * ones(size(Yscores));
            end
            
            for i = 1:size(Ypred,1)
                [~,maxIdx] = max(Yscores(i,:));
                Ypred(i,maxIdx) = 1;
            end
        end
            
        % Compute performance measure on the given outputs according to the
        % USPS dataset-specific ranking standard measure
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            
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
        
        % Scales matrix M between -1 and 1
        function Ms = scale(obj , M)
            
            mx = max(max(M));
            mn = min(min(M));
            
            Ms = ((M + abs(mn)) / (mx - mn)) * 2 - 1;
            
        end
        
        function getTrainSet(obj)
            
        end
        
        function getTestSet(obj)
            
        end
        
   end % methods
end % classdef