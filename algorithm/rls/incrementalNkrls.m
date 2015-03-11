classdef incrementalNkrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Saving options
        storeFullTrainPerf
        trainPerformance 
        storeFullValPerf
        valPerformance 
        storeFullTestPerf
        testPerformance
        
        ntr   % Number of training samples
        
        % Kernel props
        nyMapper
        mapType
        numMapParRangeSamples
        mapParGuesses
        mapParStar
        numMapParGuesses
        maxRank
        
        numNysParGuesses

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses    
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
    end
    
    methods
        
        function obj = incrementalNkrls(mapType, numMapParRangeSamples, numNysParGuesses,  numMapParGuesses , filterParGuesses , maxRank , mapParGuesses  , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
            init( obj , mapType, numMapParRangeSamples ,  numNysParGuesses, numMapParGuesses , filterParGuesses , maxRank , mapParGuesses  , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
        end
        
        function init( obj , mapType, numMapParRangeSamples ,  numNysParGuesses, numMapParGuesses , filterParGuesses , maxRank , mapParGuesses  , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
            obj.mapType = mapType;
            obj.numMapParRangeSamples = numMapParRangeSamples;
            obj.numNysParGuesses = numNysParGuesses;
            obj.numMapParGuesses = numMapParGuesses;
            obj.filterParGuesses = filterParGuesses;
            obj.maxRank = maxRank;
            obj.mapParGuesses  = mapParGuesses ;
            obj.verbose = verbose;
            obj.storeFullTrainPerf = storeFullTrainPerf;
            obj.storeFullValPerf = storeFullValPerf;
            obj.storeFullTestPerf = storeFullTestPerf;
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
            obj.nyMapper = obj.mapType(Xtrain, Ytrain , obj.ntr , 'numNysParGuesses' , obj.numNysParGuesses , 'maxRank' , obj.maxRank , 'numMapParGuesses' , obj.numMapParGuesses , 'mapParGuesses' , obj.mapParGuesses , 'filterParGuesses' , obj.filterParGuesses , 'numMapParRangeSamples' , obj.numMapParRangeSamples , 'verbose' , obj.verbose );
            obj.mapParGuesses = obj.nyMapper.rng;   % Warning: rename to mapParGuesses
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage initialization
            if obj.storeFullTrainPerf == 1
                obj.trainPerformance = zeros(size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            if obj.storeFullValPerf == 1
                obj.valPerformance = zeros(size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            if obj.storeFullTestPerf == 1
                obj.testPerformance = zeros(size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            
            while obj.nyMapper.next()
                
                obj.nyMapper.compute();
                
                % Compute Kvm TrainVal kernel
                % Initialize TrainVal kernel
                argin = {};
                argin = [argin , 'mapParGuesses' , obj.nyMapper.currentPar(2)];
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end                    
                kernelVal = obj.nyMapper.kernelType(Xval,obj.nyMapper.Xs, argin{:});
                kernelVal.next();
                kernelVal.compute();
                
                for i = 1:size(obj.filterParGuesses,2)

                    % Compute predictions matrix
                    YvalPred = kernelVal.K * obj.nyMapper.alpha{i};

                    % Compute validation performance
                    valPerf = performanceMeasure( Yval , YvalPred , valIdx );                

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %  Store performance matrices  %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    if obj.storeFullTrainPerf == 1                    
                        
                        % Initialize TrainTrain kernel
                        argin = {};
                        argin = [argin , 'mapParGuesses' , obj.nyMapper.currentPar(2)];
                        if ~isempty(obj.verbose)
                            argin = [argin , 'verbose' , obj.verbose];
                        end                    
                        kernelTrain = obj.nyMapper.kernelType(Xtrain , obj.nyMapper.Xs , argin{:});
                        kernelTrain.next();
                        kernelTrain.compute();

                        % Compute training predictions matrix
                        YtrainPred = kernelTrain.K * obj.nyMapper.alpha{i};
                        
                        % Compute training performance
                        trainPerf = performanceMeasure( Ytrain , YtrainPred , trainIdx );
                        
                        obj.trainPerformance(obj.nyMapper.currentParIdx , i) = trainPerf;
                    end
                    
                    if obj.storeFullValPerf == 1
                        obj.valPerformance(obj.nyMapper.currentParIdx , i) = valPerf;
                    end
                    
                    if obj.storeFullTestPerf == 1                    
                        
                        % Initialize TrainTest kernel
                        argin = {};
                        argin = [argin , 'mapParGuesses' , obj.nyMapper.currentPar(2)];
                        if ~isempty(obj.verbose)
                            argin = [argin , 'verbose' , obj.verbose];
                        end                    
                        kernelTest = obj.nyMapper.kernelType(Xte , obj.nyMapper.Xs , argin{:});
                        kernelTest.next();
                        kernelTest.compute();
                        
                        % Compute scores
                        YtestPred = kernelTest.K * obj.nyMapper.alpha{i};

                        % Compute training performance
                        testPerf = performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );
                        
                        obj.testPerformance(obj.nyMapper.currentParIdx , i) = testPerf;
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%
                    % Store best model %
                    %%%%%%%%%%%%%%%%%%%%
                    if valPerf < valM

                        % Update best kernel parameter combination
                        obj.mapParStar = obj.nyMapper.currentPar;

                        %Update best filter parameter
                        obj.filterParStar = obj.nyMapper.filterParGuesses(i);

                        %Update best validation performance measurement
                        valM = valPerf;

                        % Update internal model samples matrix
                        obj.Xmodel = obj.nyMapper.Xs;

                        % Update coefficients vector
                        obj.c = obj.nyMapper.alpha{i};
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
            argin = {};
            argin = [argin , 'mapParGuesses' , obj.mapParStar(2)];
            if ~isempty(obj.verbose)
                argin = [argin , 'verbose' , obj.verbose];
            end
            kernelTest = obj.nyMapper.kernelType(Xte , obj.Xmodel , argin{:});
            kernelTest.next();
            kernelTest.compute();
            
            % Compute scores
            Ypred = kernelTest.K * obj.c;
        end        
    end
end

