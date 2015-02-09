classdef incrementalNkrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Svaing options
        storeFullTrainPerf
        trainPerformance 
        storeFullValPerf
        valPerformance 
        
        ntr   % Number of training samples
        
        % Kernel props
        mapType
        kernelType
        numKerParRangeSamples
        kerParGuesses
        mapParStar
        numMapParGuesses
        fixedMapPar
        maxRank
        
        numNysParGuesses

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses    
        fixedFilterPar
        
%         trainIdx    % Training indexes used internally in the actually performed training
%         valIdx      % Validation indexes used internally in the actually
%         performed validation
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
    end
    
    methods
        
        function obj = incrementalNkrls(mapType, numKerParRangeSamples, numNysParGuesses,  numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose , storeFullTrainPerf, storeFullValPerf)
            init( obj , mapType, numKerParRangeSamples ,  numNysParGuesses, numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose , storeFullTrainPerf, storeFullValPerf)
        end
        
        function init( obj , mapType, numKerParRangeSamples ,  numNysParGuesses, numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose , storeFullTrainPerf, storeFullValPerf)
            obj.mapType = mapType;
            obj.numKerParRangeSamples = numKerParRangeSamples;
            obj.numNysParGuesses = numNysParGuesses;
            obj.numMapParGuesses = numMapParGuesses;
            obj.filterParGuesses = filterParGuesses;
            obj.maxRank = maxRank;
            obj.fixedMapPar = fixedMapPar;
            obj.verbose = verbose;
            obj.storeFullTrainPerf = storeFullTrainPerf;
            obj.storeFullValPerf = storeFullValPerf;
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
            
            obj.ntr = size(Xtrain,1);
            
            % Train kernel
%             X , Y , numNysParGuesses , numMapParGuesses , filterParGuesses , numKerParRangeSamples , maxRank , fixedMapPar , verbose)
            
            nyMapper = obj.mapType(Xtrain, Ytrain , obj.ntr , obj.numNysParGuesses , obj.numMapParGuesses , obj.filterParGuesses , obj.numKerParRangeSamples , obj.maxRank , obj.fixedMapPar , obj.verbose);
            obj.kerParGuesses = nyMapper.rng;   % Warning: rename to mapParGuesses
%             obj.filterParGuesses = [];
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage initialization
            if obj.storeFullTrainPerf == 1
                warning('Training performance matrix storage unavailable');
                obj.trainPerformance = zeros(size(obj.kerParGuesses,2), size(obj.filterParGuesses,2));
            end
            if obj.storeFullValPerf == 1
                obj.valPerformance = zeros(size(obj.kerParGuesses,2), size(obj.filterParGuesses,2));
            end

            YvalPred = zeros(size(Xval,1),size(obj.filterParGuesses,2));
            
            while nyMapper.next()
                
                nyMapper.compute();
                
                % Compute Kvm TrainVal kernel
                obj.kernelType = nyMapper.kernelType;
                kernelVal = obj.kernelType(Xval,nyMapper.Xs);
                kernelVal.compute(nyMapper.currentPar(2));
                
                for i = 1:size(obj.filterParGuesses,2)

                    % Compute predictions matrix
                    YvalPred(:,i) = kernelVal.K * nyMapper.alpha{i};

                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred(:,i) , valIdx );                

%                     if obj.storeFullTrainPerf == 1
%                         obj.trainPerformance() = trainPerf;
%                     end
                    if obj.storeFullValPerf == 1
                        obj.valPerformance(nyMapper.currentParIdx , i) = valPerf;
                    end
            
                    if valPerf < valM

                        % Update best kernel parameter combination
                        obj.mapParStar = nyMapper.currentPar;

                        %Update best filter parameter
                        obj.filterParStar = nyMapper.filterParGuesses(i);

                        %Update best validation performance measurement
                        valM = valPerf;

                        % Update internal model samples matrix
                        obj.Xmodel = nyMapper.Xs;

                        % Update coefficients vector
                        obj.c = nyMapper.alpha{i};
                    end
                end
            end
            
            % Find best parameters from validation performance matrix
            
              %[row, col] = find(valPerformance <= min(min(valPerformance)));

%             obj.kerParStar = obj.kerParGuesses
%             obj.filterParStar = ...  
            
            if obj.verbose == 1
                
                % Print best kernel hyperparameter(s)
                display('Best mapper hyperparameter(s):')
                obj.mapParStar

                % Print best filter hyperparameter(s)
                display('Best filter hyperparameter(s):')
                obj.filterParStar
                
            end
        end
        
        function Ypred = test( obj , Xte )
                
            % Get kernel type and instantiate train-test kernel (including sigma)
            kernelTest = obj.kernelType(Xte , obj.Xmodel);
            kernelTest.compute(obj.mapParStar(2));
            
            % Compute scores
            Ypred = kernelTest.K * obj.c;
            
        end
    end
end

