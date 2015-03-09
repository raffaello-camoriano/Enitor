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
        mapType
        numKerParRangeSamples
        mapParGuesses
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
        
        function obj = incrementalNkrls(mapType, numKerParRangeSamples, numNysParGuesses,  numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
            init( obj , mapType, numKerParRangeSamples ,  numNysParGuesses, numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
        end
        
        function init( obj , mapType, numKerParRangeSamples ,  numNysParGuesses, numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose , storeFullTrainPerf, storeFullValPerf, storeFullTestPerf)
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
            nyMapper = obj.mapType(Xtrain, Ytrain , obj.ntr , obj.numNysParGuesses , obj.numMapParGuesses , obj.filterParGuesses , obj.numKerParRangeSamples , obj.maxRank , obj.fixedMapPar , obj.verbose);
            obj.mapParGuesses = nyMapper.rng;   % Warning: rename to mapParGuesses
            
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
            
            while nyMapper.next()
                
                nyMapper.compute();
                
                % Compute Kvm TrainVal kernel
                kernelVal = obj.mapType(Xval,nyMapper.Xs);
                kernelVal.compute(nyMapper.currentPar(2));
                
                for i = 1:size(obj.filterParGuesses,2)

                    % Compute predictions matrix
                    YvalPred = kernelVal.K * nyMapper.alpha{i};

                    % Compute validation performance
                    valPerf = performanceMeasure( Yval , YvalPred , valIdx );                

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %  Store performance matrices  %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    if obj.storeFullTrainPerf == 1                    
                        
                        kernelTrain = obj.mapType(Xtrain,nyMapper.Xs);
                        kernelTrain.compute(nyMapper.currentPar(2));

                        % Compute training predictions matrix
                        YtrainPred = kernelTrain.K * nyMapper.alpha{i};
                        
                        % Compute training performance
                        trainPerf = performanceMeasure( Ytrain , YtrainPred , trainIdx );
                        
                        obj.trainPerformance(nyMapper.currentParIdx , i) = trainPerf;
                    end
                    
                    if obj.storeFullValPerf == 1
                        obj.valPerformance(nyMapper.currentParIdx , i) = valPerf;
                    end
                    if obj.storeFullTestPerf == 1                    
                        
                        % Instantiate train-test kernel (including sigma)
                        kernelTest = obj.mapType(Xte , nyMapper.Xs);
                        kernelTest.compute(nyMapper.currentPar(2));

                        % Compute scores
                        YtestPred = kernelTest.K * nyMapper.alpha{i};

                        % Compute training performance
                        testPerf = performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );
                        
                        obj.testPerformance(nyMapper.currentParIdx , i) = testPerf;
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%
                    % Store best model %
                    %%%%%%%%%%%%%%%%%%%%
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
            kernelTest = obj.mapType(Xte , obj.Xmodel);
            kernelTest.compute(obj.mapParStar(2));
            
            % Compute scores
            Ypred = kernelTest.K * obj.c;
        end
    end
end

