classdef krls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
                
        % I/O options
        storeFullTrainPerf  % Store full training performance matrix 1/0
        storeFullValPerf    % Store full validation performance matrix 1/0
        storeFullTestPerf   % Store full test performance matrix 1/0
        valPerformance      % Validation performance matrix
        trainPerformance    % Training performance matrix
        testPerformance     % Test performance matrix
        
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
    end
    
    methods
        
        function obj = krls(map, filter, varargin)
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
  
            % storeFullValPerf    % Store full validation performance matrix 1/0
            defaultStoreFullValPerf = 0;
            checkStoreFullValPerf = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'storeFullValPerf',defaultStoreFullValPerf,checkStoreFullValPerf);           
  
            % storeFullTestPerf   % Store full test performance matrix 1/0
            defaultStoreFullTestPerf = 0;
            checkStoreFullTestPerf = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'storeFullTestPerf',defaultStoreFullTestPerf,checkStoreFullTestPerf);            
            
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
            
            % Parse function inputs
            parse(p, map, filter, varargin{:}{:})
            
            % Assign parsed parameters to object properties
            fields = fieldnames(p.Results);
            for idx = 1:numel(fields)
                obj.(fields{idx}) = p.Results.(fields{idx});
            end
            
            %%% Joint parameters validation
            
            if isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                error('either mapParGuesses or numMapParGuesses must be specified');
            end    
            
            if ~isempty(obj.mapParGuesses) && ~isempty(obj.numMapParGuesses)
                error('mapParGuesses and numMapParGuesses cannot be specified together');
            end    
            
            if ~isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                obj.numMapParGuesses = size(obj.mapParGuesses,2);
            end
            
            if isempty(obj.filterParGuesses) && isempty(obj.numFilterParGuesses)
                error('either filterParGuesses or numFilterParGuesses must be specified');
            end         
            
            if ~isempty(obj.filterParGuesses) && ~isempty(obj.numFilterParGuesses)
                error('filterParGuesses and numFilterParGuesses cannot be specified together');
            end
            
            if ~isempty(obj.filterParGuesses) && isempty(obj.numFilterParGuesses)
                obj.numFilterParGuesses = size(obj.filterParGuesses,2);
                obj.isFilterParGuessesFixed = 1;
            else
                obj.isFilterParGuessesFixed = 0;
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
            trainIdx = 1 : ntr;
            valIdx = ntr + 1 : size(Xtr,1);
            
            Xtrain = Xtr(trainIdx,:);
            Ytrain = Ytr(trainIdx,:);
            Xval = Xtr(valIdx,:);
            Yval = Ytr(valIdx,:);
                                    
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
                obj.trainPerformance = zeros(obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            if obj.storeFullValPerf == 1
                obj.valPerformance = zeros(obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            if obj.storeFullTestPerf == 1
                obj.testPerformance = zeros(obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            
            while kernelTrain.next()
                
                % Compute kernel according to current hyperparameters
                kernelTrain.compute();

                % Initialize TrainVal kernel
                argin = {};
                argin = [argin , 'mapParGuesses' , kernelTrain.currentPar(1)];
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end                    
                kernelVal = obj.map(Xval,Xtrain, argin{:});            
                kernelVal.next();
                kernelVal.compute(kernelTrain.currentPar);
                
                % Normalization factors
                numSamples = size(Xtrain , 1);
                
                argin = {};
                if ~isempty(obj.filterParGuesses)
                    argin = [argin , 'filterParGuesses' , obj.filterParGuesses];
                end
                if ~isempty(obj.numFilterParGuesses)
                    argin = [argin , 'numFilterParGuesses' , obj.numFilterParGuesses];
                end
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end
                filter = obj.filter( kernelTrain.K, Ytrain , numSamples , argin);
                
%                 filter = obj.filter( nyMapper.C' * nyMapper.C, nyMapper.C' * Ytrain, numSamples , 'numGuesses' , obj.numFilterParGuesses, 'M' , nyMapper.W , 'fixedFilterPar' , obj.fixedFilterPar , 'verbose' , obj.verbose);
                
                obj.filterParGuessesStorage = [obj.filterParGuessesStorage ; filter.filterParGuesses];
                
                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    % Compute predictions matrix
                    YvalPred = kernelVal.K * filter.weights;
                    
                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred , valIdx );
                    
                    
                    % Populate full performance matrices
%                     trainPerformance(i,j) = performanceMeasure( kernel.K * filter.weights, Ytrain);
                                 
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %  Store performance matrices  %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if obj.storeFullTrainPerf == 1         

                        % Compute training predictions matrix
                        YtrainPred = kernelTrain.K * filter.weights;
                        
                        % Compute training performance
                        trainPerf = performanceMeasure( Ytrain , YtrainPred , trainIdx );
                        
                        obj.trainPerformance(kernelTrain.currentParIdx , filter.currentParIdx) = trainPerf;
                    end
                    
                    if obj.storeFullValPerf == 1
                        obj.valPerformance(kernelTrain.currentParIdx , filter.currentParIdx) = valPerf;
                    end
                    if obj.storeFullTestPerf == 1      
                        
                        % Initialize TrainTest kernel
                        argin = {};
                        argin = [argin , 'mapParGuesses' , kernelTrain.currentPar];
                        if ~isempty(obj.verbose)
                            argin = [argin , 'verbose' , obj.verbose];
                        end                  
                        kernelTest = obj.map(Xte , Xtrain , argin{:});
                        kernelTest.next();
                        kernelTest.compute();
                        
                        % Compute scores
                        YtestPred = kernelTest.K * filter.weights;

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
            argin = [argin , 'mapParGuesses' , obj.mapParStar];
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

