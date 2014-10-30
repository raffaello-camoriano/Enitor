classdef regls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Kernel props
        kernelType
        kerParGuesses
        kerParStar
        numKerParGuesses

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses        
        
%         trainIdx    % Training indexes used internally in the actually performed training
%         valIdx      % Validation indexes used internally in the actually performed training
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
    end
    
    methods
        
        function obj = regls(kerTy, filtTy,  numKerParGuesses , numFilterParGuesses)
            init( obj , kerTy, filtTy)
            
%             obj.numFolds = numFolds;
%             if numFolds < 2
%                 display('Minimum number of folds: 2. numFolds set to 2.')
%                 obj.numFolds = 2;
%             end

            obj.numKerParGuesses = numKerParGuesses;
            obj.numFilterParGuesses = numFilterParGuesses;
        end
        
        function init( obj , kerTy, filtTy)
            obj.kernelType = kerTy;
            obj.filterType = filtTy;
        end
        
        function lambda = computeSingleLambda( lambdas)
            lambda = median(lambdas);
        end
        
        function train(obj , Xtr , Ytr , performanceMeasure , recompute, validationPart)
            
            % Training/validation sets splitting
            shuffledIdx = randperm(size(Xtr,1));
            trainIdx = shuffledIdx(1:floor(size(Xtr,1)*(1-validationPart)));
            valIdx = shuffledIdx(ceil(size(Xtr,1)*(1-validationPart):size(Xtr,1)));
            
            Xtrain = Xtr(trainIdx,:);
            Ytrain = Ytr(trainIdx,:);
            Xval = Xtr(valIdx,:);
            Yval = Ytr(valIdx,:);
            
            % Distance matrices computation
            
            % TrainVal kernel
            kernelVal = obj.kernelType(Xval,Xtrain);
            
            % Train kernel
            kernel = obj.kernelType(Xtrain,Xtrain, obj.numKerParGuesses);
            
            % Compute kernel parameter range(s)
            kernel.range;
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage
%             trainPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
%             valPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);

            while kernel.next
                
                % Compute kernel according to current hyperparameters
                kernel.compute;
                kernelVal.compute(kernel.currentPar);
                
                filter = obj.filterType(kernel.K, Ytrain , obj.numFilterParGuesses);
                filter.range;
                
                while filter.next
                    
                    % Compute filter according to current hyperparameters
                    filter.compute;

                    % Populate full performance matrices
                    %trainPerformance(i,j) = perfm( kernel.K * filter.coeffs, Ytrain);
                    %valPerformance(i,j) = perfm( kernelVal.K * filter.coeffs, Yval);
                    
                    % Compute predictions matrix
                    YvalScores = kernelVal.K * filter.coeffs;
                    YvalPred = zeros(size(YvalScores));
                    for i = 1:size(YvalPred,1)
                        [~,maxIdx] = max(YvalScores(i,:));
                        YvalPred(i,maxIdx) = 1;
                    end
                    
                    % Compute performance
                    valPerf = performanceMeasure( YvalPred, Yval );
                    
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        % WARNING: containers.Map is a handle class. We
                        % need to instantiate obj.kerParStar by value!
                        obj.kerParStar = containers.Map(keys(kernel.currentPar),values(kernel.currentPar));
                        
                        %Update best filter parameter
                        obj.filterParStar = containers.Map(keys(filter.currentPar),values(filter.currentPar));
                        
                        %Update best validation performance measurement
                        valM = valPerf
                        
                        if ~recompute
                            
                            % Update internal model samples matrix
                            obj.Xmodel = Xtrain;
                            
                            % Update coefficients vector
                            obj.c = filter.coeffs;
                        end
                    end
                end
            end
            
            % Find best parameters from validation performance matrix
            
              %[row, col] = find(valPerformance <= min(min(valPerformance)));

%             obj.kerParStar = obj.kerParGuesses
%             obj.filterParStar = ...  
            
            % Print best kernel hyperparameter(s)
            display('Best kernel hyperparameter(s):')
            keys(obj.kerParStar)
            values(obj.kerParStar)

            % Print best filter hyperparameter(s)
            display('Best filter hyperparameter(s):')
            keys(obj.filterParStar)
            values(obj.filterParStar)
            
            if (nargin > 4) && (recompute)
                
                % Recompute kernel on the whole training set with the best
                % kernel parameter
                kernel.init(Xtr, Xtr);
                kernel.compute(obj.kerParStar);
                
                % Recompute filter on the whole training set with the best
                % filter parameter
                filter.init(kernel.K,Ytr);
                filter.compute(obj.filterParStar);
                
                % Update internal model samples matrix
                obj.Xmodel = Xtr;
                
                % Update coefficients vector
                obj.c = filter.coeffs;
            end        
        end
        
        function Ypred = test( obj , Xte )

            % Get kernel type and instantiate kernel (including sigma)
            kernelTest = obj.kernelType(Xte , obj.Xmodel , 0 , obj.kerParStar);
            
            % Compute scores
            Yscores = kernelTest.K * obj.c;
            
            % Compute predictions matrix
            Ypred = zeros(size(Yscores));
            for i = 1:size(Ypred,1)
                [~,maxIdx] = max(Yscores(i,:));
                Ypred(i,maxIdx) = 1;
            end
        end
        
        
%         function crossVal(obj , dataset)
%             
%             if obj.numFolds > dataset.nTr
%                 display(['Maximum number of folds:' dataset.nTr '. numFolds set to nTr.'])
%                 obj.numFolds = dataset.nTr;
%             end
%             
% %             if ( strcmp(obj.kernelType , 'gaussianKernel') )
% %                 
% %             else
% %                 error('Kernel type not implemented.');
% %             end
% 
%             % Performance measure storage variable
%             perfMeas = zeros(obj.numFolds , obj.numKerParGuesses , obj.numFilterParGuesses);
%             
%             % Get kernel type and instantiate kernel
%             if strcmp(obj.kernelType , 'gaussian');
%                 obj.kernel = gaussianKernel( dataset.X , dataset.X );
%             end
%             
%             % Get guesses vector for the kernel parameter
%             kerParGuesses = obj.kernel.range(obj.numKerParGuesses);
%             
%             % Get guesses vector for the regularization filter parameter
%             %filterParGuesses = obj.filter.range(obj.kernel , obj.numFilterParGuesses); % TODO: implement filter range
%             filterParGuesses = 1:obj.numFilterParGuesses;
%             
%             for i = 1:obj.numFolds
%                 for j = 1:obj.numKerParGuesses
%                     for k = 1:obj.numFilterParGuesses
%                         
%                         i
%                         j
%                         k
% 
%                         kerPar = obj.kerParGuesses(j);
%                         filterPar = obj.filterParGuesses(k);
%                         
%                         testFoldIdx = round(dataset.nTr/obj.numFolds)*(i-1) + 1 : round(dataset.nTr/obj.numFolds)*i;                
%                         trainFoldIdx = setdiff(dataset.trainIdx, testFoldIdx);
% 
%                         obj.train(dataset.X(trainFoldIdx,:) , dataset.Y(trainFoldIdx), kerPar, filterPar);
%                         Ypred = obj.test( dataset.X(trainFoldIdx,:) , dataset.X(testFoldIdx,:) , kerPar);
% 
%                         perfMeas(i,j,k) = dataset.performanceMeasure(dataset.Y(testFoldIdx) , Ypred);
% 
%                     end
%                 end
%             end
%             
%             % Set the index of the best kernel parameter
%             [~,kerParStarIdx] = min(median(perfMeas,1),2);
%             obj.kerParStar = kerParGuesses(kerParStarIdx);
%                         
%             % Set the index of the best filter parameter
%             [~,filterParStarIdx] = min(median(perfMeas,1),3);
%             obj.filterParStar = filterParGuesses(filterParStarIdx);
%             
%             % Compute the kernel with the best kernel parameter
%             obj.kernel.compute(obj.kerParStar);
%             
%         end

    end
end

