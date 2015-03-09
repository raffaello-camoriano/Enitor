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
        
        function obj = krls(kerTy, filtTy,  numMapParGuesses , numFilterParGuesses , verbose)
            init( obj , kerTy, filtTy ,  numMapParGuesses , numFilterParGuesses , verbose)
        end
        
        function init( obj , kerTy, filtTy , numMapParGuesses , numFilterParGuesses , verbose)
            obj.kernelType = kerTy;
            obj.filterType = filtTy;
            obj.numMapParGuesses = numMapParGuesses;
            obj.numFilterParGuesses = numFilterParGuesses;
            obj.verbose = verbose;
        end
        
        function train(obj , Xtr , Ytr , performanceMeasure , recompute, validationPart)
            
            % Training/validation sets splitting
%             shuffledIdx = randperm(size(Xtr,1));
            ntr = floor(size(Xtr,1)*(1-validationPart));
%             trainIdx = shuffledIdx(1 : tmp1);
%             valIdx = shuffledIdx(tmp1 + 1 : end);
            trainIdx = 1 : ntr;
            valIdx = ntr + 1 : size(Xtr,1);
            
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
                
                filter = obj.filterType( kernel.K, Ytrain , numSamples , 'numGuesses' , obj.numFilterParGuesses , 'verbose' , obj.verbose);
                
%                 filter = obj.filterType( nyMapper.C' * nyMapper.C, nyMapper.C' * Ytrain, numSamples , 'numGuesses' , obj.numFilterParGuesses, 'M' , nyMapper.W , 'fixedFilterPar' , obj.fixedFilterPar , 'verbose' , obj.verbose);
                
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
%             semilogx(cell2mat(filter.rng),  valPerformance);            
            
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

