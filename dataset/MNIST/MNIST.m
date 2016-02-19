
classdef MNIST < dataset
   
   properties
        outputFormat

        classes
        trainClassFreq 
        testClassFreq 
   end
   
   methods
        function obj = MNIST(nTr , nTe, outputFormat , shuffleTraining, shuffleTest, shuffleAll, varargin)
            
            % Call superclass constructor with arguments
            obj = obj@dataset([], shuffleTraining, shuffleTest, shuffleAll);
            
%            Optional parameters: classes, trainClassFreq, testClassFreq            

            if ~isempty(varargin) && ~isempty(varargin{1}{1}) 
                classes = varargin{1}{1};
            else
                classes = 0:9;
                display('Classes set to 0 to 9 by default');
            end

            if ~isempty(varargin) && ~isempty(varargin{1}{2}) 
                obj.trainClassFreq = varargin{1}{2};
            else
                obj.trainClassFreq = 1/numel(classes) * ones(size(classes));
                display('Balanced training classes');
            end

            if ~isempty(varargin) && ~isempty(varargin{1}{3}) 
                obj.testClassFreq = varargin{1}{3};
            else
                obj.testClassFreq = 1/numel(classes) * ones(size(classes));
                display('Balanced test classes');
            end

            % Set t (number of classes)
            classes = unique(classes);
            obj.classes = classes;
            obj.t = numel(classes);
            
            % Set output format
            if isempty(outputFormat)
                error('outputFormat not set');
            end
            if (strcmp(outputFormat, 'zeroOne') || ...
                    strcmp(outputFormat, 'plusMinusOne') || ...
                    strcmp(outputFormat, 'plusOneMinusBalanced'))
                obj.outputFormat = outputFormat;
            else
                display('Wrong or missing output format, set to plusMinusOne by default');
                obj.outputFormat = 'plusMinusOne';
            end
            
            
            data = load('mnist_all.mat');
            
            gnd = [];
            
            for i = obj.classes
                currentFieldStr = strcat('train' , num2str(i));
                obj.X = [ obj.X ; data.(currentFieldStr) ];
                gnd = [gnd ; i * ones(size(data.(currentFieldStr),1) , 1) ];
            end
                
            obj.nTrTot = size(obj.X,1);
            
            for i = obj.classes
                currentFieldStr = strcat('test' , num2str(i));
                obj.X = [ obj.X ; data.(currentFieldStr) ];
                gnd = [gnd ; i * ones(size(data.(currentFieldStr),1) , 1) ];
            end
            
            obj.X = double(obj.X);
            obj.X = obj.scale(obj.X);
            
            obj.n = size(obj.X , 1);
            obj.nTeTot = obj.n - obj.nTrTot;
            
            obj.d = size(obj.X , 2);
            obj.t = numel(obj.classes);
            

            % Find class indices
            classIdxTr = cell(1,obj.t);
            classIdxTe = cell(1,obj.t);
            for i = 1:obj.t
                classIdxTr{i} = find(gnd(1:obj.nTrTot) == classes(i));
                classIdxTe{i} = obj.nTrTot + find(gnd(obj.nTrTot+1:obj.n) == classes(i));
            end


            % Compute number of points for each class (for training and
            % test sets
            trainClassNum = zeros(1,obj.t);
            testClassNum = zeros(1,obj.t);
            for i = 1:obj.t    
                trainClassNum(i) = ...
                    round((obj.trainClassFreq(i) * numel(classIdxTr{i}))/max(obj.trainClassFreq));
                
                testClassNum(i) = ...
                    round((obj.testClassFreq(i) * numel(classIdxTe{i}))/max(obj.testClassFreq));
            end
                
            % Subsample training set
            obj.trainIdx =[];
            if isempty(nTr)
                obj.nTr = sum(trainClassNum);
            else
                obj.nTr = nTr;        
            end
            
            if obj.nTr > sum(trainClassNum)
                error('nTr > obj.nTrTot');
            end

            sumTrainClassNum = sum(trainClassNum);
            for i = 1:obj.t                       
                trainClassNum(i) = round(trainClassNum(i) / sumTrainClassNum  * obj.nTr);
                obj.trainIdx = [obj.trainIdx ; classIdxTr{i}(randperm( numel(classIdxTr{i}), trainClassNum(i)))];
            end
            
            
            % Subsample test set
            obj.testIdx =[];
            if isempty(nTe)
                obj.nTe = sum(testClassNum);
            else
                obj.nTe = nTe;        
            end
            
            if obj.nTe > sum(testClassNum)
                error('nTe > obj.nTeTot');
            end

            sumTestClassNum = sum(testClassNum);
            for i = 1:obj.t                       
                testClassNum(i) = round(testClassNum(i) / sumTestClassNum  * obj.nTe);
                obj.testIdx = [obj.testIdx ; classIdxTe{i}(randperm( numel(classIdxTe{i}), testClassNum(i)))];
            end
            
            
            % Reformat output columns
            if (nargin > 2) && (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
                obj.outputFormat = outputFormat;
            else
                display('Wrong or missing output format, set to plusMinusOne by default');
                obj.outputFormat = 'plusMinusOne';
            end

            
            if obj.t == 2
                if strcmp(obj.outputFormat, 'zeroOne')
                    obj.Y = zeros(obj.n,1);
                elseif strcmp(obj.outputFormat, 'plusMinusOne')
                    obj.Y = - ones(obj.n,1);
                elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                    obj.Y = - ones(obj.n,1)/(obj.t-1);
                end
                obj.Y(gnd == obj.classes(1),1) = 1;
            else
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
            end            
            
            
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
            
            % Set problem type
            obj.problemType = 'classification';
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
            Ypred = obj.scoresToClasses(Ypred);
            
            
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