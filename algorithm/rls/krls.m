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
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
    end
    
    methods
        
        function obj = krls(kerTy, filtTy,  numMapParGuesses , numFilterParGuesses )
            init( obj , kerTy, filtTy ,  numMapParGuesses , numFilterParGuesses )
        end
        
        function init( obj , kerTy, filtTy , numMapParGuesses , numFilterParGuesses )
            obj.kernelType = kerTy;
            obj.filterType = filtTy;
            obj.numMapParGuesses = numMapParGuesses;
            obj.numFilterParGuesses = numFilterParGuesses;
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
                        
            % TrainVal kernel
            kernelVal = obj.kernelType(Xval,Xtrain);

            % Train kernel
            kernel = obj.kernelType(Xtrain,Xtrain, obj.numMapParGuesses);
            obj.kerParGuesses = kernel.rng;
            obj.filterParGuesses = [];
            
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage
            trainPerformance = zeros(1, 25);
            valPerformance = [];

            
            while kernel.next()
                
                % Compute kernel according to current hyperparameters
                kernel.compute();
                kernelVal.compute(kernel.currentPar);
                
                % Normalization factors
                numSamples = size(Xtrain , 1);
                
                filter = obj.filterType( kernel.K, Ytrain , numSamples , obj.numFilterParGuesses);
                obj.filterParGuesses = [obj.filterParGuesses ; filter.rng];

                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();


                    % Compute predictions matrix
                    YvalPred = kernelVal.K * filter.weights;
                    
                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred , valIdx );
                    
                    
                    % Populate full performance matrices
%                     trainPerformance(i,j) = performanceMeasure( kernel.K * filter.weights, Ytrain);
                    valPerformance = [valPerformance valPerf];
                                 
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
                            obj.c = filter.weights;
                        end
                    end
                end
            end
            
            
            % Plot errors
            semilogx(cell2mat(filter.rng),  valPerformance);            
            
            
            
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
                numSamples = size(Xtr , 1);

                filter.init( kernel.K , Ytr , numSamples);
                filter.compute(obj.filterParStar);
                
                % Update internal model samples matrix
                obj.Xmodel = Xtr;
                
                % Update coefficients vector
                obj.c = filter.weights;
            end        
        end
        
        function Ypred = test( obj , Xte )

            % Get kernel type and instantiate train-test kernel (including sigma)
            kernelTest = obj.kernelType(Xte , obj.Xmodel);
            kernelTest.compute(obj.kerParStar);
            
            % Compute scores
            Ypred = kernelTest.K * obj.c;
        end
    end
end

