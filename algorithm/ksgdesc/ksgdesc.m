classdef ksgdesc < algorithm
    %KGDESC Kernelized gradient descent estimator/algorithm
    %   Also known as Landwaeber algorithm, the kernelized gradient descent
    %   estimator/algorithm optimizes the coefficients c of the expression:
    %   
    %   $f(x) = \Sigma_{c = 1..n} K(x_i,x)c_i$
    %
    
    properties
                
        % I/O options
        storeFullTrainPerf  % Store full training performance matrix 1/0
        storeFullTrainPred  % Store full training predictions matrix 1/0
        storeFullValPerf    % Store full validation performance matrix 1/0
        storeFullTestPerf   % Store full test performance matrix 1/0
        storeFullTestPred   % Store full test predictions matrix 1/0
        
        valPerformance      % Validation performance matrix
        trainPerformance    % Training performance matrix
        trainPred           % Training predictions matrix
        testPerformance     % Test performance matrix
        testPred            % Test predictions matrix
        
        % Kernel props
        map
        mapParGuesses
        mapParStar
        numMapParGuesses

        % Filter props
        filter
        filterParStar
        filterParGuesses
        numFilterParGuesses       
        filterParGuessesStorage
        isFilterParGuessesFixed
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
        
        initialWeights  % Initial filter weights
        eta         % filter step size
        theta       % filter step size sequence exponent
        
        trainIdx % Actual training indexes
        valIdx   % Actual validation indexes
        
        % Stopping rule
        stoppingRule        % Handle to the stopping rule
        
    end
    
    methods
        
        function obj = ksgdesc(map, filter, varargin)
            init( obj , map, filter, varargin)
        end
        
        function init( obj , map, filter , varargin)

            p = inputParser;
            
            %%%% Required parameters
            
            % map
            checkMap = @(x) isa(x,'function_handle');
            addRequired(p,'map',checkMap);

            % filter
            checkFilter = @(x) isa(x,'function_handle');
            addRequired(p,'filter', checkFilter);
            
            %%%% Optional parameters

            % verbose             % 1: verbose; 0: silent      
            defaultVerbose = 0;
            checkVerbose = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'verbose',defaultVerbose,checkVerbose);

            % storeFullTrainPerf  % Store full training performance matrix 1/0
            defaultStoreFullTrainPerf = 0;
            checkStoreFullTrainPerf = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'storeFullTrainPerf',defaultStoreFullTrainPerf,checkStoreFullTrainPerf);           
            
            % storeFullTrainPred   % Store full test predictions matrix 1/0
            defaultStoreFullTrainPred = 0;
            checkStoreFullTrainPred = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'storeFullTrainPred',defaultStoreFullTrainPred,checkStoreFullTrainPred);            
            
            % storeFullValPerf    % Store full validation performance matrix 1/0
            defaultStoreFullValPerf = 0;
            checkStoreFullValPerf = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'storeFullValPerf',defaultStoreFullValPerf,checkStoreFullValPerf);           
  
            % storeFullTestPerf   % Store full test performance matrix 1/0
            defaultStoreFullTestPerf = 0;
            checkStoreFullTestPerf = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'storeFullTestPerf',defaultStoreFullTestPerf,checkStoreFullTestPerf);            
            
            % storeFullTestPred   % Store full test predictions matrix 1/0
            defaultStoreFullTestPred = 0;
            checkStoreFullTestPred = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'storeFullTestPred',defaultStoreFullTestPred,checkStoreFullTestPred);            
            
            % mapParGuesses       % Map parameter guesses cell array
            defaultMapParGuesses = [];
            checkMapParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'mapParGuesses',defaultMapParGuesses,checkMapParGuesses);                    
            
            % numMapParGuesses        % Number of map parameter guesses
            defaultNumMapParGuesses = [];
            checkNumMapParGuesses = @(x) x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses);        
            
            % filterParGuesses       % filter parameter guesses vector
            defaultFilterParGuesses = [];
            checkFilterParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'filterParGuesses',defaultFilterParGuesses,checkFilterParGuesses);                
            
            % numFilterParGuesses    % Number of filter parameter guesses vector
            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x)  x > 0 ;            
            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses,checkNumFilterParGuesses);            
            
            % initialWeights    % initial weights of the filter
            defaultInitialWeights = [];
            addParameter(p,'initialWeights',defaultInitialWeights);      
            
            % eta    % step size
            defaultEta = [];
            addParameter(p,'eta',defaultEta);              
            
            % theta    % Exponent of step size decreasing sequence
            defaultTheta = [];
            checkTheta = @(x)  (x <= 1 && x >= 0);            
            addParameter(p,'theta',defaultTheta,checkTheta);
            
            % stoppingRule
            defaultStoppingRule = [];
            checkStoppingRule = @(x) isobject(x);
            addParameter(p,'stoppingRule', defaultStoppingRule , checkStoppingRule);            
            
            % Parse function inputs
            parse(p, map, filter, varargin{:}{:})
            
            % Assign parsed parameters to object properties
            fields = fieldnames(p.Results);
            for idx = 1:numel(fields)
                obj.(fields{idx}) = p.Results.(fields{idx});
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
            ntr = floor(size(Xtr,1)*(1-validationPart));
%             trainIdx = shuffledIdx(1 : tmp1);
%             valIdx = shuffledIdx(tmp1 + 1 : end);
            obj.trainIdx = 1 : ntr;
            obj.valIdx = ntr + 1 : size(Xtr,1);
            
            Xtrain = Xtr(obj.trainIdx,:);
            Ytrain = Ytr(obj.trainIdx,:);
            Xval = Xtr(obj.valIdx,:);
            Yval = Ytr(obj.valIdx,:);
                                    
            % Initialize Train kernel
            argin = {};
            if ~isempty(obj.numMapParGuesses)
                argin = [argin , 'numMapParGuesses' , obj.numMapParGuesses];
            end
            if ~isempty(obj.mapParGuesses)
                argin = [argin , 'mapParGuesses' , obj.mapParGuesses];
            end
            if ~isempty(obj.verbose)
                argin = [argin , 'verbose' , obj.verbose];
            end
            kernelTrain = obj.map( Xtrain , Xtrain , argin{:});
            
            obj.mapParGuesses = kernelTrain.mapParGuesses;   % Warning: rename to mapParGuesses
            obj.filterParGuessesStorage = [];

            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage initialization
            if obj.storeFullTrainPerf == 1
                obj.trainPerformance = NaN*zeros(obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            if obj.storeFullTrainPred == 1
                obj.trainPred = zeros(obj.numMapParGuesses, obj.numFilterParGuesses,size(Xtrain,1));
            end
            if obj.storeFullValPerf == 1
                obj.valPerformance = NaN*zeros(obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            if obj.storeFullTestPerf == 1
                obj.testPerformance = NaN*zeros(obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            if obj.storeFullTestPred == 1
                obj.testPred = zeros(obj.numMapParGuesses, obj.numFilterParGuesses,size(Xte,1));
            end
            
            while kernelTrain.next()
                
                % Compute kernel according to current hyperparameters
                kernelTrain.compute();
                
                % Initialize TrainVal kernel
                argin = {};
                argin = [argin , 'mapParGuesses' , full(kernelTrain.currentPar(1))];
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end                    
                kernelVal = obj.map(Xval,Xtrain, argin{:});            
                kernelVal.next();
                kernelVal.compute(kernelTrain.currentPar);
                
                % Normalization factors
                numSamples = size(Xtrain , 1);
                
                if isempty(obj.numFilterParGuesses)
                    obj.numFilterParGuesses = size(Xtrain,1);
                end
                argin = {};
                argin = [argin , 'numFilterParGuesses' , obj.numFilterParGuesses];
                if ~isempty(obj.initialWeights)
                    argin = [argin , 'initialWeights' , obj.initialWeights];
                end
                if ~isempty(obj.eta)
                    argin = [argin , 'eta' , obj.eta];
                end
                if ~isempty(obj.theta)
                    argin = [argin , 'theta' , obj.theta];
                end
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end
                filter = obj.filter( obj.map , kernelTrain.currentPar(1) , Xtrain , Ytrain , numSamples , argin{:});
                
                obj.filterParGuessesStorage = [obj.filterParGuessesStorage ; filter.filterParGuesses];
                
                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    % Compute predictions matrix
                    YvalPred = kernelVal.K * filter.weights;

                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred ,obj.valIdx );
                    
                    % Apply early stopping criterion
                    stop = 0;
                    if ~isempty(obj.stoppingRule)
                        stop = obj.stoppingRule.evaluate(valPerf);
                    end
                    
                    if stop == 1
                        break;
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %  Store performance matrices  %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if obj.storeFullTrainPerf == 1         

                        % Compute training predictions matrix
                        YtrainPred = kernelTrain.K * filter.weights;
                        if obj.storeFullTrainPred == 1
                            obj.trainPred(kernelTrain.currentParIdx , filter.currentParIdx,:) = YtrainPred;
                        end
                        
                        % Compute training performance
                        trainPerf = performanceMeasure( Ytrain , YtrainPred , obj.trainIdx );
                        
                        obj.trainPerformance(kernelTrain.currentParIdx , filter.currentParIdx) = trainPerf;
                    end
                    
                    if obj.storeFullValPerf == 1
                        obj.valPerformance(kernelTrain.currentParIdx , filter.currentParIdx) = valPerf;
                    end
                    if obj.storeFullTestPerf == 1      
                        
                        % Initialize TrainTest kernel
                        argin = {};
                        argin = [argin , 'mapParGuesses' , full(kernelTrain.currentPar)];
                        if ~isempty(obj.verbose)
                            argin = [argin , 'verbose' , obj.verbose];
                        end                  
                        kernelTest = obj.map(Xte , Xtrain , argin{:});
                        kernelTest.next();
                        kernelTest.compute();
                        
                        % Compute scores
                        YtestPred = kernelTest.K * filter.weights;
                        if obj.storeFullTestPred == 1
                            obj.testPred(kernelTrain.currentParIdx , filter.currentParIdx,:) = YtestPred;
                        end

                        % Compute training performance
                        testPerf = performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );
                        
                        obj.testPerformance(kernelTrain.currentParIdx , filter.currentParIdx) = testPerf;                        
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%
                    % Store best model %
                    %%%%%%%%%%%%%%%%%%%%
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        obj.mapParStar = kernelTrain.currentPar;
                        
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
            obj.mapParStar

            % Print best filter hyperparameter(s)
            display('Best filter hyperparameter(s):')
            obj.filterParStar
            
            if (nargin > 4) && (recompute)
                
                % Recompute kernel on the whole training set with the best
                % kernel parameter
                kernelTrain.init(Xtr, Xtr);
                kernelTrain.compute(obj.mapParStar);
                
                % Recompute filter on the whole training set with the best
                % filter parameter
                numSamples = size(Xtr , 1);

                filter.init( kernelTrain.K , Ytr , numSamples);
                filter.compute(obj.filterParStar);
                
                % Update internal model samples matrix
                obj.Xmodel = Xtr;
                
                % Update coefficients vector
                obj.c = filter.weights;
            end        
        end
        
        function Ypred = test( obj , Xte )

            % Get kernel type and instantiate train-test kernel (including sigma)
            argin = {};
            argin = [argin , 'mapParGuesses' , full(obj.mapParStar)];
            if ~isempty(obj.verbose)
                argin = [argin , 'verbose' , obj.verbose];
            end
            kernelTest = obj.map(Xte , obj.Xmodel , argin{:});
            kernelTest.next();
            kernelTest.compute();

            % Compute scores
            Ypred = kernelTest.K * obj.c;
        end
    end
end

