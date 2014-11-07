classdef lrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses        
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
    end
    
    methods
        
        function obj = lrls(filtTy, numFilterParGuesses)
            init( obj , kerTy, filtTy)
            
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
            tmp1 = floor(size(Xtr,1)*(1-validationPart));
            trainIdx = shuffledIdx(1 : tmp1);
            valIdx = shuffledIdx(tmp1 + 1 : end);
            
            Xtrain = Xtr(trainIdx,:);
            Ytrain = Ytr(trainIdx,:);
            Xval = Xtr(valIdx,:);
            Yval = Ytr(valIdx,:);
            
            % Distance matrices computation
            
            % TrainVal kernel
            kernelVal = obj.kernelType(Xval,Xtrain);
            
            % Train kernel
            kernel = obj.kernelType(Xtrain,Xtrain, obj.numKerParGuesses);
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage
%             trainPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
%             valPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);

            while kernel.next()
                
                % Compute kernel according to current hyperparameters
                kernel.compute();
                kernelVal.compute(kernel.currentPar);
                
                filter = obj.filterType(kernel.K, Ytrain , obj.numFilterParGuesses);
                
                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    % Populate full performance matrices
                    %trainPerformance(i,j) = perfm( kernel.K * filter.coeffs, Ytrain);
                    %valPerformance(i,j) = perfm( kernelVal.K * filter.coeffs, Yval);
                    
                    % Compute predictions matrix
                    YvalPred = kernelVal.K * filter.coeffs;
                    
                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred );
                    
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        obj.kerParStar = kernel.currentPar;
                        
                        %Update best filter parameter
                        obj.filterParStar = filter.currentPar;
                        
                        %Update best validation performance measurement
                        valM = valPerf;
                        
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
            obj.kerParStar

            % Print best filter hyperparameter(s)
            display('Best filter hyperparameter(s):')
            obj.filterParStar
            
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
            kernelTest = obj.kernelType(Xte , obj.Xmodel);
            kernelTest.compute(obj.kerParStar);
            
            % Compute scores
            Ypred = kernelTest.K * obj.c;

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

