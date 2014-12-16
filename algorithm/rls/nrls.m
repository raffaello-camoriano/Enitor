classdef nrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Kernel props
        mapType
        kernelType
        numKerParRangeSamples
        kerParGuesses
        mapParStar
        numMapParGuesses
        maxRank

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
        
        function obj = nrls(mapType, numKerParRangeSamples, filterType,  numMapParGuesses , numFilterParGuesses , maxRank)
            init( obj , mapType, numKerParRangeSamples, filterType ,  numMapParGuesses , numFilterParGuesses , maxRank)
        end
        
        function init( obj , mapType, numKerParRangeSamples , filterType , numMapParGuesses , numFilterParGuesses , maxRank)
            obj.mapType = mapType;
            obj.numKerParRangeSamples = numKerParRangeSamples;
            obj.filterType = filterType;
            obj.numMapParGuesses = numMapParGuesses;
            obj.numFilterParGuesses = numFilterParGuesses;
            obj.maxRank = maxRank;
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

            % Train kernel
            nyMapper = obj.mapType(Xtrain, obj.numMapParGuesses , obj.numKerParRangeSamples , obj.maxRank);
            obj.kerParGuesses = nyMapper.rng;
            obj.filterParGuesses = [];
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage
%             trainPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
%             valPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);

            while nyMapper.next()
                
                % Compute kernel according to current hyperparameters
                nyMapper.compute();
                                        
                % Compute Kvm TrainVal kernel
                obj.kernelType = nyMapper.kernelType;
                kernelVal = nyMapper.kernelType(Xval,nyMapper.Xs);
                kernelVal.compute(nyMapper.currentPar(2));
                
                % Normalization factors
                numSamples = size(Xtrain , 1);
                
                filter = obj.filterType( nyMapper.C' * nyMapper.C, Ytrain(nyMapper.sampledPoints,:) , numSamples , obj.numFilterParGuesses, nyMapper.W);
                obj.filterParGuesses = [obj.filterParGuesses ; filter.rng];

                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    % Populate full performance matrices
                    %trainPerformance(i,j) = perfm( kernel.K * filter.weights, Ytrain);
                    %valPerformance(i,j) = perfm( kernelVal.K * filter.weights, Yval);
                    
                    % Compute predictions matrix
                    YvalPred = kernelVal.K * filter.weights;
                    
                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred , valIdx );
                    
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        obj.mapParStar = nyMapper.currentPar;
                        
                        %Update best filter parameter
                        obj.filterParStar = filter.currentPar;
                        
                        %Update best validation performance measurement
                        valM = valPerf;
                        
                        if ~recompute
                            
                            % Update internal model samples matrix
                            obj.Xmodel = nyMapper.Xs;
                            
                            % Update coefficients vector
                            obj.c = filter.weights;
                        end
                    end
                end
            end
            
            % Find best parameters from validation performance matrix
            
              %[row, col] = find(valPerformance <= min(min(valPerformance)));

%             obj.kerParStar = obj.kerParGuesses
%             obj.filterParStar = ...  
            
            % Print best kernel hyperparameter(s)
            display('Best mapper hyperparameter(s):')
            obj.mapParStar

            % Print best filter hyperparameter(s)
            display('Best filter hyperparameter(s):')
            obj.filterParStar
            
            %nyMapper = obj.mapType(Xtrain, obj.numMapParGuesses , obj.numKerParRangeSamples , obj.maxRank);
                        
            if (nargin > 4) && (recompute)
                
                % Recompute kernel on the whole training set with the best
                % kernel parameter
                nyMapper.init(Xtr, Xtr);
                nyMapper.compute(obj.kerParStar);
                
                % Recompute filter on the whole training set with the best
                % filter parameter
                numSamples = size(Xtr , 1);

                filter.init( nyMapper.K , Ytr , numSamples);
                filter.compute(obj.filterParStar);
                
                % Update internal model samples matrix
                obj.Xmodel = nyMapper.Xs;
                
                % Update coefficients vector
                obj.c = filter.weights;
            end
        end
        
        function Ypred = test( obj , Xte )

                                                    
%                 % Compute Kvm TrainVal kernel
%                 kernelVal = nyMapper.kernelType(Xval,nyMapper.Xs);
%                 kernelVal.compute(nyMapper.currentPar(2));
                
            % Get kernel type and instantiate train-test kernel (including sigma)
%             kernelTest = nyMapper.kernelType(Xte , obj.Xmodel);
            kernelTest = obj.kernelType(Xte , obj.Xmodel);
            kernelTest.compute(obj.mapParStar(2));
            
            % Compute scores
            Ypred = kernelTest.K * obj.c;
        end
    end
end

