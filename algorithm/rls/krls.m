classdef krls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Kernel props
        kernelType
        kerParGuesses
        kerParStar
        numMapParGuesses

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses        
        
%         trainIdx    % Training indexes used internally in the actually performed training
%         valIdx      % Validation indexes used internally in the actually
%         performed validation
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
    end
    
    methods
        
        function obj = krls(kerTy, filtTy,  numMapParGuesses , numFilterParGuesses)
            init( obj , kerTy, filtTy)
            
%             obj.numFolds = numFolds;
%             if numFolds < 2
%                 display('Minimum number of folds: 2. numFolds set to 2.')
%                 obj.numFolds = 2;
%             end

            obj.numMapParGuesses = numMapParGuesses;
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
            kernel = obj.kernelType(Xtrain,Xtrain, obj.numMapParGuesses);
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage
%             trainPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
%             valPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);

            while kernel.next()
                
                % Compute kernel according to current hyperparameters
                kernel.compute();
                kernelVal.compute(kernel.currentPar);
                
                filter = obj.filterType( 'dual' , kernel.K, Ytrain , obj.numFilterParGuesses);
                
                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    % Populate full performance matrices
                    %trainPerformance(i,j) = perfm( kernel.K * filter.coeffs, Ytrain);
                    %valPerformance(i,j) = perfm( kernelVal.K * filter.coeffs, Yval);
                    
                    % Compute predictions matrix
                    YvalPred = kernelVal.K * filter.coeffs;
                    
                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred , valIdx );
                    
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
                filter.init('dual' , kernel.K , Ytr);
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
    end
end

