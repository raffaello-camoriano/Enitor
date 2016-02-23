
classdef iCubWorld28 < dataset
   
   properties
        outputFormat
        classes
        trainClassFreq 
        testClassFreq 
        dataRoot
        trainFolder 
        testFolder
        trainClassNum
        testClassNum
   end
   
   methods
        function obj = iCubWorld28(nTr , nTe, outputFormat , shuffleTraining, shuffleTest, shuffleAll, varargin)
            
            % Call superclass constructor with arguments
            obj = obj@dataset([], shuffleTraining, shuffleTest, shuffleAll);
            
%            Optional parameters: classes, trainClassFreq, testClassFreq, dataRoot, trainFolder, testFolder

            obj.nTr = nTr;
            obj.nTe = nTe;
            
            if ~isempty(varargin) && ~isempty(varargin{1}{1}) 
                classes = varargin{1}{1};
            else
                classes = 1:28;
                display('Classes set to 1 to 28 by default');
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

            if ~isempty(varargin) && ~isempty(varargin{1}{4}) 
                obj.dataRoot = varargin{1}{4};
            else
                obj.dataRoot = 'data/caffe_centralcrop_meanimagenet2012/';
                display(['dataRoot set to: ' , obj.dataRoot]);
            end

            if ~isempty(varargin) && ~isempty(varargin{1}{5}) 
                obj.trainFolder = varargin{1}{5};
            else
                obj.trainFolder = 'lunedi22';
                display(['trainFolder set to: ' , obj.trainFolder]);
            end
                
            if ~isempty(varargin) && ~isempty(varargin{1}{6}) 
                obj.testFolder = varargin{1}{6};
            else                
                obj.testFolder = 'martedi23';
                display(['testFolder set to: ' , obj.testFolder]);
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
            
            % Load data from folders
            Xtr = [];
            Ytr = [];
            Xte = [];
            Yte = [];
            for i = 1:numel(obj.trainFolder)
                [Xtr1, Ytr1, Xte1, Yte1] = loadIcub28([], ...
                                                [], ...
                                                obj.classes, ...
                                                obj.trainClassFreq, ...
                                                obj.testClassFreq, ...
                                                obj.dataRoot, ...
                                                obj.trainFolder{i}, ...
                                                obj.testFolder{i});                        
                Xtr = [Xtr ; Xtr1];
                Ytr = [Ytr ; Ytr1];
                Xte = [Xte ; Xte1];
                Yte = [Yte ; Yte1];
            end
            clear Xtr1 Ytr1 Xte1 Yte1;
            
            obj.nTrTot = size(Xtr,1);
            obj.nTeTot = size(Xte,1);
            if isempty(obj.nTr)
                obj.nTr = obj.nTrTot;
            end
            if isempty(obj.nTe)
                obj.nTe = obj.nTeTot;
            end
            obj.n = obj.nTrTot + obj.nTeTot ;
            obj.d = size(Xtr , 2);
            
            % Select consecutive samples
            obj.trainIdx = 1:obj.nTr;          
            obj.testIdx = obj.nTrTot + 1 : obj.nTrTot + obj.nTe;     
            
            obj.X = [Xtr ; Xte];
            Ytmp = [Ytr ; Yte];
            
            % Compute trainClassNum and testClassNum
            [~,gnd] = max(Ytmp(obj.trainIdx,:),[],2);
            for k = 1:obj.t
                obj.trainClassNum(k) = sum(gnd==k);
            end
            [~,gnd] = max(Ytmp(obj.testIdx,:),[],2);
            for k = 1:obj.t
                obj.testClassNum(k) = sum(gnd==k);
            end
            clear Xtr Xte Ytr Yte

            if obj.t == 2
                if strcmp(obj.outputFormat, 'zeroOne')
                    obj.Y = zeros(obj.n,1);
                elseif strcmp(obj.outputFormat, 'plusMinusOne')
                    obj.Y = - ones(obj.n,1);
                elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                    obj.Y = - ones(obj.n,1)/(obj.t-1);
                end
                obj.Y(Ytmp(:,obj.classes(1)) == 1,1) = 1;
            else
                if strcmp(obj.outputFormat, 'zeroOne')
                    obj.Y = (Ytmp + 1) / 2;
                elseif strcmp(obj.outputFormat, 'plusMinusOne')
                    obj.Y = Ytmp;
                elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                    obj.Y = ((Ytmp - 1) * 2/(obj.t - 1) * ones(obj.n,obj.t)) + 1;
                end
            end
            
            clear Ytmp
                                            
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
            
            if numel(obj.classes) == 2
                if strcmp(obj.outputFormat, 'zeroOne')
                    Ypred = zeros(size(Yscores));
                elseif strcmp(obj.outputFormat, 'plusMinusOne')
                    Ypred = -1 * ones(size(Yscores));
                elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                    Ypred = -1/(obj.t - 1) * ones(size(Yscores));
                end
                Ypred(Yscores>0) = 1;
            else
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
        end
            
        % Compute performance measure on the given outputs according to the
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            
            
            % Check if Ypred is real-valued. If yes, convert it.
%             Ypred = obj.scoresToClasses(Yscores);
% 
%             perf = obj.lossFunction(Y, Yscores, Ypred);
            
            
            if size(obj.classes,2) == 2
                Ypred = obj.scoresToClasses(Ypred);
                correctIdx = Y == Ypred;
                numCorrect = sum(correctIdx);
                perf = 1 - (numCorrect / size(Y,1));
            else
                Ypred = obj.scoresToClasses(Ypred);
            
                C = transpose(bsxfun(@eq, Y', Ypred'));
                D = sum(C,2);
                E = D == size(Y,2);
                numCorrect = sum(E);
                perf = 1 - (numCorrect / size(Y,1));     

            end
        end
        
        % Scales matrix M between 0 and 1
        function Ms = scale(obj , M)
            
            Ms = M/255;

%             mx = max(M);
%             mn = min(M);
% 
%             Ms = ((M - mn) / (mx - mn));
        end        
   end % methods
end % classdef