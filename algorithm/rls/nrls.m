classdef nrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        ntr
        
        storeFullTrainPerf
        storeFullValPerf
        valPerformance
        trainPerformance
        storeFullTestPerf
        testPerformance
        
        numNysParGuesses
        
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
        fixedFilterParGuesses
        numFilterParGuesses    
        fixedMapParGuesses
        
%         trainIdx    % Training indexes used internally in the actually performed training
%         valIdx      % Validation indexes used internally in the actually
%         performed validation
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
    end
    
    methods
        
        function obj = nrls(mapType, numKerParRangeSamples, filterType, numNysParGuesses , numMapParGuesses , numFilterParGuesses , maxRank , fixedMapParGuesses , fixedFilterParGuesses , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
            init( obj , mapType, numKerParRangeSamples, filterType , numNysParGuesses , numMapParGuesses , numFilterParGuesses , maxRank , fixedMapParGuesses , fixedFilterParGuesses , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
        end
        
        function init( obj , mapType, numKerParRangeSamples , filterType , numNysParGuesses , numMapParGuesses , numFilterParGuesses , maxRank , fixedMapParGuesses , fixedFilterParGuesses , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
            obj.mapType = mapType;
            obj.numKerParRangeSamples = numKerParRangeSamples;
            obj.filterType = filterType;
            obj.numNysParGuesses = numNysParGuesses;
            obj.numMapParGuesses = numMapParGuesses;
            obj.numFilterParGuesses = numFilterParGuesses;
            obj.maxRank = maxRank;
            obj.fixedMapParGuesses = fixedMapParGuesses;
            obj.fixedFilterParGuesses = fixedFilterParGuesses;
            obj.verbose = verbose;
            obj.storeFullTrainPerf = storeFullTrainPerf;
            obj.storeFullValPerf = storeFullValPerf;
            obj.storeFullTestPerf = storeFullTestPerf;
            
            if obj.numFilterParGuesses ~= size(obj.fixedFilterParGuesses,2)
                error('obj.numFilterParGuesses ~= size(obj.fixedFilterParGuesses,2)');
            end
        end
        
        function train(obj , Xtr , Ytr , performanceMeasure , recompute, validationPart , varargin)
                        
            p = inputParser;
            
            %%%% Required parameters
            
            checkRecompute = @(x) x == 1 || x == 0 ;
            checkValidationPart = @(x) x > 0 && x < 1;
            
            addRequired(p,'Xtr');
            addRequired(p,'Ytr');
            addRequired(p,'performanceMeasure');
            addRequired(p,'recompute',checkRecompute);
            addRequired(p,'validationPart',checkValidationPart);
            
            %%%% Optional parameters
            % Optional parameter names:
            % Xte, Yte
            
            defaultXte = [];
            checkXte = @(x) size(x,2) == size(Xtr,2);
            
            defaultYte = [];
            checkYte = @(x) size(x,2) == size(Ytr,2);
            
            addParameter(p,'Xte',defaultXte,checkXte)
            addParameter(p,'Yte',defaultYte,checkYte)

            % Parse function inputs
            parse(p, Xtr , Ytr , performanceMeasure , recompute, validationPart , varargin{:})
            
            Xte = p.Results.Xte;
            Yte = p.Results.Yte;
            
            % Training/validation sets splitting
%             shuffledIdx = randperm(size(Xtr,1));
            tmp1 = floor(size(Xtr,1)*(1-validationPart));
%             trainIdx = shuffledIdx(1 : tmp1);
%             valIdx = shuffledIdx(tmp1 + 1 : end);
            trainIdx = 1 : tmp1;
            valIdx = tmp1 + 1 : size(Xtr,1);
            
            Xtrain = Xtr(trainIdx,:);
            Ytrain = Ytr(trainIdx,:);
            Xval = Xtr(valIdx,:);
            Yval = Ytr(valIdx,:);

            obj.ntr = size(Xtrain,1);

            % Train kernel
            nyMapper = obj.mapType(Xtrain, obj.numNysParGuesses , obj.numMapParGuesses , obj.numKerParRangeSamples , obj.maxRank , obj.fixedMapParGuesses , obj.verbose);
            obj.kerParGuesses = nyMapper.rng;   % Warning: rename to mapParGuesses
%             obj.filterParGuesses = [];
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage initialization
            if obj.storeFullTrainPerf == 1
                obj.trainPerformance = zeros(size(obj.kerParGuesses,2), obj.numFilterParGuesses);
            end
            if obj.storeFullValPerf == 1
                obj.valPerformance = zeros(size(obj.kerParGuesses,2), obj.numFilterParGuesses);
            end
            if obj.storeFullTestPerf == 1
                obj.testPerformance = zeros(size(obj.kerParGuesses,2), obj.numFilterParGuesses);
            end

            while nyMapper.next()
                
                % Compute kernel according to current hyperparameters
                nyMapper.compute();
                                        
                % Compute Kvm TrainVal kernel
                obj.kernelType = nyMapper.kernelType;
                kernelVal = obj.kernelType(Xval,nyMapper.Xs);
                kernelVal.compute(nyMapper.currentPar(2));
                
%                 if ~isempty(obj.fixedFilterParGuesses)
%                     filter = obj.filterType( nyMapper.C' * nyMapper.C, nyMapper.C' * Ytrain, numSamples , 'numGuesses' , obj.numFilterParGuesses, 'M' , nyMapper.W , 'fixedFilterParGuesses' , obj.fixedFilterParGuesses , 'verbose' , obj.verbose);
%                 else
%                     filter = obj.filterType( nyMapper.C' * nyMapper.C, nyMapper.C' * Ytrain, numSamples , 'numGuesses' , obj.numFilterParGuesses, 'M' , nyMapper.W , 'verbose' , obj.verbose);
%                 end                
                if ~isempty(obj.fixedFilterParGuesses)
                    filter = obj.filterType( nyMapper.C' * nyMapper.C, nyMapper.C' * Ytrain, obj.ntr , 'numGuesses' , obj.numFilterParGuesses, 'M' , eye(size(nyMapper.W,1)) , 'fixedFilterParGuesses' , obj.fixedFilterParGuesses , 'verbose' , obj.verbose);
                else
                    filter = obj.filterType( nyMapper.C' * nyMapper.C, nyMapper.C' * Ytrain, obj.ntr , 'numGuesses' , obj.numFilterParGuesses, 'M' , eye(size(nyMapper.W,1)) , 'verbose' , obj.verbose);
                end
                
                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    % Compute predictions matrix
                    YvalPred = kernelVal.K * filter.weights;

                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred , valIdx );
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %  Store performance matrices  %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    if obj.storeFullTrainPerf == 1                    
                        
                        kernelTrain = obj.kernelType(Xtrain,nyMapper.Xs);
                        kernelTrain.compute(nyMapper.currentPar(2));

                        % Compute training predictions matrix
                        YtrainPred = kernelTrain.K * filter.weights;
                        
                        % Compute training performance
                        trainPerf = performanceMeasure( Ytrain , YtrainPred , trainIdx );
                        
                        obj.trainPerformance(nyMapper.currentParIdx , filter.currentParIdx) = trainPerf;
                    end
                    
                    if obj.storeFullValPerf == 1
                        obj.valPerformance(nyMapper.currentParIdx , filter.currentParIdx) = valPerf;
                    end
                    if obj.storeFullTestPerf == 1                    
                        
                        % Instantiate train-test kernel (including sigma)
                        kernelTest = obj.kernelType(Xte , nyMapper.Xs);
                        kernelTest.compute(nyMapper.currentPar(2));

                        % Compute scores
                        YtestPred = kernelTest.K * filter.weights;

                        % Compute training performance
                        testPerf = performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );
                        
                        obj.testPerformance(nyMapper.currentParIdx , filter.currentParIdx) = testPerf;
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%
                    % Store best model %
                    %%%%%%%%%%%%%%%%%%%%
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        obj.mapParStar = nyMapper.currentPar;
                        
                        %Update best filter parameter
                        obj.filterParStar = filter.currentPar;
                        
                        %Update best validation performance measurement
                        valM = valPerf;
                        
                        % Update internal model samples matrix
                        obj.Xmodel = nyMapper.Xs;

                        % Update coefficients vector
                        obj.c = filter.weights;
                    end
                end
            end
            
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

