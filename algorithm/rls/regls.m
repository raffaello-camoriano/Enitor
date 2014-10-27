classdef regls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kernelType
        kerParGuesses
        kerParStar
        filterType
        filterParStar
        filterParGuesses
        c
        numFolds
        numKerParGuesses
        numFilterParGuesses
    end
    
    methods
        
        function obj = regls(kerTy, filtTy, numFolds , numKerParGuesses , numFilterParGuesses)
            init( obj , kerTy, filtTy)
            
            obj.numFolds = numFolds;
            if numFolds < 2
                display('Minimum number of folds: 2. numFolds set to 2.')
                obj.numFolds = 2;
            end

            obj.numKerParGuesses = numKerParGuesses;
            obj.numFilterParGuesses = numFilterParGuesses;
        end
        
        function init( obj , kerTy, filtTy)
            obj.kernelType = kerTy;
            obj.filterType = filtTy;
        end
        
        function lambda = computeSingleLambda(obj , lambdas)
            lambda = median(lambdas);
        end
        
        function train(obj , X , Y, recompute, validationPart)
            
            % Training/validation sets splitting
            shuffledIdx = randperm(size(X,1));
            trainIdx = shuffledIdx(1:floor(size(X,1)*(1-validationPart)));
            valIdx = shuffledIdx(ceil(size(X,1)*(1-validationPart):size(X,1)));
            
            Xtrain = X(trainIdx,:);
            Ytrain = Y(trainIdx,:);
            Xval = X(valIdx,:);
            Yval = Y(valIdx,:);
            
            % Distance matrices computation
            
            % TrainVal kernel
            kernelVal = obj.kernelType;
            kernelVal.init(Xval,Xtrain);
            
            % Train kernel
            kernel = obj.kernelType;
            kernel.init(Xtrain,Xtrain);
            obj.kerParGuesses = kernel.range(5);
            
            valM = inf;     % Keeps track of the lowes validation error
            
            % Full matrices for performance storage
%             trainPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
%             valPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
            
            for kerPar = obj.kerParGuesses
                
                kernel.compute(kerPar);
                kernelVal.compute(kerPar);
                
                %filter = obj.filterType(kernel.K);
                filter = obj.filterType;
                filter.init(kernel.K, Ytrain);
                filter.range(5);
                obj.filterParGuesses = filter.rng;
                
                for filterPar = obj.filterParGuesses
                    filter.compute(filterPar, Ytrain);

                    % Populate full performance matrices
                    %trainPerformance(i,j) = perfm( kernel.K * filter.coeffs, Ytrain);
                    %valPerformance(i,j) = perfm( kernelVal.K * filter.coeffs, Yval);
                    
                    if perfm( kernelVal.K * filter.coeffs, Yval) < valM
                        
                        % Update best kernel parameter
                        obj.kerParStar = kerPar;
                        
                        %Update best filter parameter
                        obj.filterParStar = filterPar;
                        
                        %Update best validation performance measurement
                        valM = perfm( kernelVal.K * filter.coeffs, Yval);
                        
                        % Update coefficients vector
                        obj.c = filter.coeffs;
                    end
                end
            end
            
            
                
            % Find best parameters from validation performance matrix
            
              [row, col] = find(valPerformance <= min(min(valPerformance)));

%             obj.kerParStar = obj.kerParGuesses
%             obj.filterParStar = ...  
            
            if argin > 3 && recompute
                
                % Recompute kernel on the whole training set with the best
                % kernel parameter
                kernel.init(X, X);
                kernel.compute(obj.kerParStar);
                
                % Recompute filter on the whole training set with the best
                % filter parameter
                filter.init(kernel.K);
                filter.compute(obj.filterParStar, Y);
                
                % Update coefficients vector
                obj.c = filter.coeffs;
            end        
        end
        
        function [Ypred , testPerformance] = test( obj , Xtr, Xte , sigma)

            % Get kernel type and instantiate kernel (including sigma)
            kernelTest = obj.kernelType(Xte,Xtrain);
            kernelTest.compute(obj.kerParStar);

            % Perform predictions
            Ypred = trainTestKer.K * obj.c;
            
            % Compute performance
            testPerformance = perfm(Ypred, Yval);
            
        end
        
        
        function crossVal1(obj , dataset)
            
            if obj.numFolds > dataset.nTr
                display(['Maximum number of folds:' dataset.nTr '. numFolds set to nTr.'])
                obj.numFolds = dataset.nTr;
            end
            
%             if ( strcmp(obj.kernelType , 'gaussianKernel') )
%                 
%             else
%                 error('Kernel type not implemented.');
%             end

            % Performance measure storage variable
            perfMeas = zeros(obj.numFolds , obj.numKerParGuesses , obj.numFilterParGuesses);
            
            % Get kernel type and instantiate kernel
            if strcmp(obj.kernelType , 'gaussian');
                obj.kernel = gaussianKernel( dataset.X , dataset.X );
            end
            
            % Get guesses vector for the kernel parameter
            kerParGuesses = obj.kernel.range(obj.numKerParGuesses);
            
            % Get guesses vector for the regularization filter parameter
            %filterParGuesses = obj.filter.range(obj.kernel , obj.numFilterParGuesses); % TODO: implement filter range
            filterParGuesses = 1:obj.numFilterParGuesses;
            
            for i = 1:obj.numFolds
                for j = 1:obj.numKerParGuesses
                    for k = 1:obj.numFilterParGuesses
                        
                        i
                        j
                        k

                        kerPar = kerParGuesses(j);
                        filterPar = filterParGuesses(k);
                        
                        testFoldIdx = round(dataset.nTr/obj.numFolds)*(i-1) + 1 : round(dataset.nTr/obj.numFolds)*i;                
                        trainFoldIdx = setdiff(dataset.trainIdx, testFoldIdx);

                        obj.train(dataset.X(trainFoldIdx,:) , dataset.Y(trainFoldIdx), kerPar, filterPar);
                        Ypred = obj.test( dataset.X(trainFoldIdx,:) , dataset.X(testFoldIdx,:) , kerPar);

                        perfMeas(i,j,k) = dataset.performanceMeasure(dataset.Y(testFoldIdx) , Ypred);

                    end
                end
            end
            
            % Set the index of the best kernel parameter
            [~,kerParStarIdx] = min(median(perfMeas,1),2);
            obj.kerParStar = kerParGuesses(kerParStarIdx);
                        
            % Set the index of the best filter parameter
            [~,filterParStarIdx] = min(median(perfMeas,1),3);
            obj.filterParStar = filterParGuesses(filterParStarIdx);
            
            % Compute the kernel with the best kernel parameter
            obj.kernel.compute(obj.kerParStar);
            
        end
    end
end

