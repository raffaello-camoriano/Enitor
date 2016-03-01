
classdef Cifar10 < dataset
   
   properties
        outputFormat
        classes
%         t
%         Y
   end
   
   methods
        function obj = Cifar10(nTrPerClass , nTePerClass, outputFormat , classes)
            
            if nTrPerClass > 5000 || nTePerClass > 1000
                error('Too many samples per class required')
            end
            
            if isempty(classes)
                error('Specify classes vector');
            end

            % Set output format
            if isempty(outputFormat)
                error('outputFormat not set');
            end
            if (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
                obj.outputFormat = outputFormat;
            else
                display('Wrong or missing output format, set to plusMinusOne by default');
                obj.outputFormat = 'plusMinusOne';
            end            
            
            % Set t (number of classes)
            classes = unique(classes);
            obj.classes = classes;
            obj.t = numel(classes);
            
            if  obj.t == 10
                % Load training batches
                if ~isempty(nTrPerClass)
                    numBatches = ceil(nTrPerClass*obj.t/10000);
                else
                    numBatches = 5;
                end
                for i  = 1:numBatches
                    traindata{i} = load(['data_batch_',num2str(i),'.mat']);
                end

                gnd = [];
                obj.X = [];
                for i = 1:numBatches
                    obj.X = [ obj.X ; traindata{i}.data ];
                    gnd = [gnd ; traindata{i}.labels];
                end
                traindata = [];

                obj.nTr = nTrPerClass * numel(classes);
                obj.nTrTot = size(obj.X,1);

                % Load test batches
                testdata = load('test_batch.mat');
                obj.X = [ obj.X ; testdata.data ];
                gnd = [gnd ; testdata.labels];
                testdata = [];

                obj.X = double(obj.X);
                obj.X = obj.scale(obj.X);

                obj.n = size(obj.X , 1);
                obj.nTe = nTePerClass * numel(classes);
                obj.nTeTot = obj.n - obj.nTrTot;
                
                obj.d = size(obj.X , 2);

                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr + 1 : obj.nTr + obj.nTe;

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
                
            else

                obj.Y = [];
                gnd = [];
                obj.X = [];
                                
                obj.nTeTot = 1000 * numel(classes);
                obj.nTrTot = 5000* numel(classes);
                obj.nTot =  obj.nTrTot +  obj.nTeTot;
                obj.nTr = nTrPerClass * numel(classes);
                obj.nTe = nTePerClass * numel(classes);
                obj.n =  obj.nTr +  obj.nTe;
                
                % Load class-specific files
                for i = classes
                    
                    % Training
                    load(['Cifar10_training_' , num2str(i) , '.mat'] )
                    obj.X = [ obj.X ; Xtr(1:nTrPerClass,:) ];
                    gnd = [gnd ; Ytr(1:nTrPerClass)];
                end
                for i = classes
                    
                    % Test
                    load(['Cifar10_test_' , num2str(i) , '.mat'] )
                    obj.X = [ obj.X ; Xte(1:nTePerClass,:) ];
                    gnd = [ gnd ; Yte(1:nTePerClass)];
                end
                
                % Scale data
                obj.X = double(obj.X);
                obj.X = obj.scale(obj.X);
                
                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr + 1 : obj.nTr + obj.nTe;
                obj.d = size(obj.X , 2);
                    
                if numel(classes) == 2
                    if strcmp(obj.outputFormat, 'zeroOne')
                        obj.Y = zeros(size(gnd,1),1);
                        obj.Y(gnd == classes(2),1) = 1;
                    elseif strcmp(obj.outputFormat, 'plusMinusOne')
                        obj.Y = ones(size(gnd,1),1);
                        obj.Y(gnd == classes(2),1) = -1;
                    elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                        obj.Y = ones(size(gnd,1),1)/(obj.t-1);
                        obj.Y(gnd == classes(2),1) = -1;
                    end

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
            end
            
            obj.shuffleTrainIdx();
            obj.shuffleTestIdx();         
            
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
            
            if numel(obj.classes) == 2
                if strcmp(obj.outputFormat, 'zeroOne')
                    Ypred = zeros(size(Yscores));
                elseif strcmp(obj.outputFormat, 'plusMinusOne')
                    Ypred = -1 * ones(size(Yscores));
                elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                    Ypred = -1/(obj.t - 1) * ones(size(Yscores));
                end

    %             for i = 1:size(Ypred,1)
    %                 if Yscores(i) > 0
    %                     Ypred(i) = 1;
    %                 end
    %             end

                Ypred(Yscores>0) = 1;
            else
                if strcmp(obj.outputFormat, 'zeroOne')
                    Ypred = zeros(size(Yscores));
                elseif strcmp(obj.outputFormat, 'plusMinusOne')
                    Ypred = -1 * ones(size(Yscores));
                elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                    Ypred = -1/(obj.t - 1) * ones(size(Yscores));
                end

%                 for i = 1:size(Ypred,1)
%                     [~,maxIdx] = max(Yscores(i,:));
%                     Ypred(i,maxIdx) = 1;
%                 end
                [~,maxIdx] = max(Yscores , [] , 2);
                Ypred(:,maxIdx) = 1;
            end
        end
            
        % Compute performance measure on the given outputs according to the
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            
            if size(obj.classes,2) == 2
                Ypred = obj.scoresToClasses(Ypred);
                correctIdx = Y == Ypred;
                numCorrect = sum(correctIdx);
                perf = 1 - (numCorrect / size(Y,1));
            else
                Ypred = obj.scoresToClasses(Ypred);

                % Compute error rate
                numCorrect = 0;
                for i = 1:size(Y,1)
                    if sum(Y(i,:) == Ypred(i,:)) == size(Y,2)
                        numCorrect = numCorrect +1;
                    end
                end

                perf = 1 - (numCorrect / size(Y,1));     
                
%                 C = transpose(bsxfun(@eq, Y', Ypred'));
%                 D = sum(C,2);
%                 E = D == size(Y,2);
%                 numCorrect = sum(E);
%                 perf = 1 - (numCorrect / size(Y,1));     

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