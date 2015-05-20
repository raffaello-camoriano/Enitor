classdef incrementalrfrls < algorithm
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
        storeFullTrainTime  % Store full training time matrix 1/0
        trainTime           % Training time matrix

        
        ntr   % Number of training samples
        
        % Mapped data
        XValTilda
        XTestTilda
        
        
        % Kernel props
        rfMapper
        mapType
        numMapParRangeSamples
        mapParGuesses
        mapParStar
        numMapParGuesses
        minRank
        maxRank
        
        numRFParGuesses

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses    
        
        XrfStar                 % Best mapping of the training set
        rfOmegaStar             % Best random omega matrix
        rfBStar                 % Best coefficients vector b
        
        w           % Weights vector        
        
        % Stopping rule
        stoppingRule        % Handle to the stopping rule
    end
    
    methods
        
        function obj = incrementalrfrls(mapType , maxRank , varargin)
            init( obj , mapType, maxRank , varargin)
        end
        
        function init( obj , mapType, maxRank , varargin)

            display('Note that incrementalrfrls can only use the Tikhonov filter in this implementation.');
            p = inputParser;
            
            %%%% Required parameters
            
            checkMaxRank = @(x) x > 0 ;

            addRequired(p,'mapType');
            addRequired(p,'maxRank',checkMaxRank);
            
            %%%% Optional parameters
            % Optional parameter names:

            defaultMinRank = 1;            
            checkMinRank = @(x) x > 0 ;
            addParameter(p,'minRank',defaultMinRank,checkMinRank);                    
            
            defaultNumRFParGuesses = 1;            
            checkNumRFParGuesses = @(x) x > 0 ;
            addParameter(p,'numRFParGuesses',defaultNumRFParGuesses,checkNumRFParGuesses);                    
            
            % mapParGuesses       % Map parameter guesses cell array
            defaultMapParGuesses = [];
            checkMapParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'mapParGuesses',defaultMapParGuesses,checkMapParGuesses);    
            
            % numMapParGuesses        % Number of map parameter guesses
            defaultNumMapParGuesses = [];
            checkNumMapParGuesses = @(x) x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses); 
            
            % numMapParRangeSamples        % Number of samples used for map
            % optimal map parameter range generation
            defaultNumMapParRangeSamples = [];            
            checkNumMapParRangeSamples = @(x) x > 0 ;
            addParameter(p,'numMapParRangeSamples',defaultNumMapParRangeSamples,checkNumMapParRangeSamples);                    
            
            % filterParGuesses       % Filter parameter guesses cell array
            defaultfFilterParGuesses = [];
            checkFilterParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'filterParGuesses',defaultfFilterParGuesses,checkFilterParGuesses);    
                   
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
            
            % storeFullTrainTime  % Store full training time matrix 1/0
            defaultStoreFullTrainTime = 0;
            checkStoreFullTrainTime = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'storeFullTrainTime',defaultStoreFullTrainTime,checkStoreFullTrainTime);            
            
            % verbose             % 1: verbose; 0: silent      
            defaultVerbose = 0;
            checkVerbose = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'verbose',defaultVerbose,checkVerbose);
    
            % stoppingRule
            defaultStoppingRule = [];
            checkStoppingRule = @(x) isobject(x);
            addParameter(p,'stoppingRule', defaultStoppingRule , checkStoppingRule);            
            
            
            % Parse function inputs
            if isempty(varargin{:})
                parse(p, mapType , maxRank )
            else
                parse(p, mapType , maxRank ,  varargin{:}{:})
            end
            
            % Assign parsed parameters to object properties
            fields = fieldnames(p.Results);
%             fieldsToIgnore = {'X1','X2'};
%             fields = setdiff(fields, fieldsToIgnore);
            for idx = 1:numel(fields)
                obj.(fields{idx}) = p.Results.(fields{idx});
            end
            
            %%% Joint parameters validation
            
            if obj.minRank > obj.maxRank
                error('The specified minimum rank of the kernel approximation is larger than the maximum one.');
            end 
            
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
            
            % Initialize Random Features Mapper
            argin = {};
            if ~isempty(obj.numRFParGuesses)
                argin = [argin , 'numRFParGuesses' , obj.numRFParGuesses];
            end      
            if ~isempty(obj.maxRank)
                argin = [argin , 'maxRank' , obj.maxRank];
            end      
            if ~isempty(obj.minRank)
                argin = [argin , 'minRank' , obj.minRank];
            end      
            if ~isempty(obj.numMapParGuesses)
                argin = [argin , 'numMapParGuesses' , obj.numMapParGuesses];
            end      
            if ~isempty(obj.mapParGuesses)
                argin = [argin , 'mapParGuesses' , full(obj.mapParGuesses)];
            end      
%             if ~isempty(obj.filterParGuesses)
%                 argin = [argin , 'filterParGuesses' , obj.filterParGuesses];
%             end           
            if ~isempty(obj.numMapParRangeSamples)
                argin = [argin , 'numMapParRangeSamples' , obj.numMapParRangeSamples];
            end     
            if ~isempty(obj.verbose)
                argin = [argin , 'verbose' , obj.verbose];
            end
            obj.rfMapper = obj.mapType(Xtrain, Ytrain , obj.ntr , argin{:} );
            obj.mapParGuesses = obj.rfMapper.rng;   % Warning: rename to mapParGuesses
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage initialization
            if obj.storeFullTrainPerf == 1
                obj.trainPerformance = NaN*zeros(size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            if obj.storeFullValPerf == 1
                obj.valPerformance = NaN*zeros(size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            if obj.storeFullTestPerf == 1
                obj.testPerformance = NaN*zeros(size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            if obj.storeFullTrainTime == 1
                obj.trainTime = NaN*zeros(size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            
            for i = 1:size(obj.filterParGuesses,2)
                
                obj.rfMapper.resetPar();
                if ~isempty(obj.stoppingRule)
                    obj.stoppingRule.reset();
                end
                obj.rfMapper.filterParGuesses = obj.filterParGuesses(i);
                
                while obj.rfMapper.next()

                    if obj.storeFullTrainTime == 1
                        tic
                    end
                    
                    obj.rfMapper.compute();

                    if obj.storeFullTrainTime == 1 && ((isempty(obj.rfMapper.prevPar) && obj.rfMapper.currentParIdx == 1) || (~isempty(obj.rfMapper.prevPar) && obj.rfMapper.currentPar(1) < obj.rfMapper.prevPar(1)))
                        obj.trainTime(obj.rfMapper.currentParIdx , i) = toc;
                    elseif obj.storeFullTrainTime == 1
                        obj.trainTime(obj.rfMapper.currentParIdx , i) = obj.trainTime(obj.rfMapper.currentParIdx - 1 , i) + toc;
                    end
                    
                    obj.XValTilda = obj.rfMapper.map(Xval);
                    
                    % Compute predictions matrix
                    YvalPred = obj.XValTilda * obj.rfMapper.alpha{1};

                    % Compute validation performance
                    valPerf = performanceMeasure( Yval , YvalPred , valIdx );                

                    % Apply early stopping criterion
                    stop = 0;
                    if ~isempty(obj.stoppingRule)
                        stop = obj.stoppingRule.evaluate(valPerf);
                    end

                    if stop == 1
                        obj.rfMapper.resetPar();
                        break;
                    end

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %  Store performance matrices  %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if obj.storeFullTrainPerf == 1                    

                        % Compute training predictions matrix
                        YtrainPred = obj.rfMapper.Xrf * obj.rfMapper.alpha{1};

                        % Compute validation performance
                        trainPerf = performanceMeasure( Ytrain , YtrainPred , trainIdx );                

                        obj.trainPerformance(obj.rfMapper.currentParIdx , i) = trainPerf;
                    end

                    if obj.storeFullValPerf == 1
                        obj.valPerformance(obj.rfMapper.currentParIdx , i) = valPerf;
                    end

                    if obj.storeFullTestPerf == 1                    

                        obj.XTestTilda = obj.rfMapper.map(Xte);

                        % Compute predictions matrix
                        YtestPred = obj.XTestTilda * obj.rfMapper.alpha{1};

                        % Compute validation performance
                        testPerf = performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );                

                        obj.testPerformance(obj.rfMapper.currentParIdx , i) = testPerf;
                    end

                    %%%%%%%%%%%%%%%%%%%%
                    % Store best model %
                    %%%%%%%%%%%%%%%%%%%%
                    if valPerf < valM

                        % Update best kernel parameter combination
                        obj.mapParStar = obj.rfMapper.currentPar;

                        % Update best filter parameter
                        obj.filterParStar = obj.rfMapper.filterParGuesses;

                        % Update best validation performance measurement
                        valM = valPerf;

                        % Update coefficients vector
                        obj.w = obj.rfMapper.alpha{1};
                    
                        % Update best mapped samples
                        obj.XrfStar = obj.rfMapper.Xrf;
                        
                        % Update best projections matrix
                        obj.rfOmegaStar = obj.rfMapper.omega;
                        
                        % Update bestb coefficients
                        obj.rfBStar = obj.rfMapper.b;
                    end
                end
            end
            
            % Free memory
            obj.rfMapper.M = [];
            obj.rfMapper.alpha = [];
            
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
            
            % Set best omega
            obj.rfMapper.omega = obj.rfOmegaStar;
            
            % Set best b
            obj.rfMapper.b = obj.rfBStar;
            
            % Set best mapping parameters
            obj.rfMapper.currentPar = obj.mapParStar;
            
            % Map test data
            XteRF = obj.rfMapper.map(Xte);

            % Compute predictions matrix
            Ypred = XteRF * obj.w;
        end        
    end
end

