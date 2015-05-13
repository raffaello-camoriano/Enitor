classdef rfrls < algorithm
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
        
        ntr   % Number of training samples
        
        maxRank     % Maximum number of RF

        % Feature mapping props
        mapType
        mapParGuesses
        mapParStar
        numMapParGuesses
        rfMapper
        XrfStar                 % Best mapping of the training set
        rfOmegaStar             % Best random omega matrix
        rfBStar                 % Best coefficients vector b
        numMapParRangeSamples   % Number of samples used for kernel hyperparameter range guesses creation
        maxNumRF                % Maximum number of random features to be used
        numRFParGuesses
        
        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses
        filter
        
        filterParGuessesStorage
        
        w               % Weights vector
        XValTilda
    end
    
    methods
        
        function obj = rfrls(mapType , filterType , maxRank , varargin)
            init( obj , mapType, filterType , maxRank , varargin)
%             init( obj , mapTy , numKerParRangeSamples , filtTy,  numMapParGuesses , numFilterParGuesses , maxRank)
        end
        
        function init( obj , mapType, filterType , maxRank , varargin)

            p = inputParser;
            
            %%%% Required parameters
            
            checkMaxRank = @(x) x > 0 ;

            addRequired(p,'mapType');
            addRequired(p,'filterType');
            addRequired(p,'maxRank',checkMaxRank);
            
            %%%% Optional parameters
            % Optional parameter names:

%             defaultNumRFParGuesses = 1;            
%             checkNumRFParGuesses = @(x) x > 0 ;
%             addParameter(p,'numRFParGuesses',defaultNumRFParGuesses,checkNumRFParGuesses);                    
            
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
            
            % verbose             % 1: verbose; 0: silent      
            defaultVerbose = 0;
            checkVerbose = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'verbose',defaultVerbose,checkVerbose);
    
            
            % Parse function inputs
            if isempty(varargin{:})
                parse(p, mapType , filterType ,  maxRank )
            else
                parse(p, mapType , filterType , maxRank ,  varargin{:}{:})
            end
            
            % Assign parsed parameters to object properties
            fields = fieldnames(p.Results);
%             fieldsToIgnore = {'X1','X2'};
%             fields = setdiff(fields, fieldsToIgnore);
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
            end                    
        end
        
        function train(obj , Xtr , Ytr , performanceMeasure , recompute, validationPart , varargin)
            
            % Training/validation sets splitting
%             shuffledIdx = randperm(size(Xtr,1));
            tmp1 = floor(size(Xtr,1)*(1-validationPart));
%             trainIdx = shuffledIdx(1 : tmp1);
%             valIdx = shuffledIdx(tmp1 + 1 : end);
            trainIdx = 1 : tmp1;
            valIdx = tmp1 + 1 : size(Xtr,1); 
                
            Xtrain = Xtr(trainIdx,:);
            Xval = Xtr(valIdx,:);    
            Ytrain = Ytr(trainIdx,:);
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
            if ~isempty(obj.numMapParGuesses)
                argin = [argin , 'numMapParGuesses' , obj.numMapParGuesses];
            end      
            if ~isempty(obj.mapParGuesses)
                argin = [argin , 'mapParGuesses' , full(obj.mapParGuesses)];
            end      
            if ~isempty(obj.numMapParRangeSamples)
                argin = [argin , 'numMapParRangeSamples' , obj.numMapParRangeSamples];
            end     
            if ~isempty(obj.verbose)
                argin = [argin , 'verbose' , obj.verbose];
            end
            obj.rfMapper = obj.mapType(Xtr, Ytr , obj.ntr , argin{:} );
            
            % mapper instantiation
%             obj.rfMapper = obj.mapType(Xtrain , Ytrain , numel(trainIdx) , obj.numMapParGuesses , obj.numKerParRangeSamples , obj.maxNumRF);
            obj.mapParGuesses = obj.rfMapper.rng;
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

            while obj.rfMapper.next()
                
                % Map samples with new hyperparameters
                obj.rfMapper.compute();
                
                % Get mapped samples according to the new map parameters
                % combination
                Xtrain = obj.rfMapper.Xrf(trainIdx,:);
                Xval = obj.rfMapper.Xrf(valIdx,:);
%                 obj.rfMapper.Xrf;
%                 obj.XValTilda = obj.rfMapper.map(Xval);
                    
                    
                % Compute covariance matrix of the training samples
                C = Xtrain' * Xtrain;
                
                % Normalization factors
%                 numSamples = size(obj.rfMapper.Xrf , 1);
                numSamples = size(Xtrain , 1);
                
%                 obj.filter = obj.filterType( C  , Xtrain' * Ytrain , numSamples ,  'numFilterParGuesses' ,  obj.numFilterParGuesses);
                
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
                filter = obj.filterType( C, Xtrain' * Ytrain , numSamples , argin{:});
                                
                
                obj.filterParGuessesStorage = [obj.filterParGuessesStorage ; filter.filterParGuesses];
                
                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    % Populate full performance matrices
                    %trainPerformance(i,j) = perfm( kernel.K * obj.filter.coeffs, Ytrain);
                    %valPerformance(i,j) = perfm( kernelVal.K * obj.filter.coeffs, Yval);
                    
                    % Compute predictions matrix
                    YvalPred = obj.XValTilda * filter.weights;
                    
                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred );
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %  Store performance matrices  %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    if obj.storeFullTrainPerf == 1                    

                        % Compute predictions matrix
                        YtrainPred = obj.rfMapper.Xrf * filter.weights;

                        % Compute validation performance
                        trainPerf = performanceMeasure( Ytrain , YtrainPred , trainIdx );                

                        obj.trainPerformance(obj.rfMapper.currentParIdx , filter.currentParIdx) = trainPerf;
                    end

                    if obj.storeFullValPerf == 1
                        obj.valPerformance(obj.rfMapper.currentParIdx , filter.currentParIdx) = valPerf;
                    end

                    if obj.storeFullTestPerf == 1                    

                        obj.XTestTilda = obj.rfMapper.map(Xtest);

                        % Compute predictions matrix
                        YtestPred = obj.XTestTilda * filter.weights;

                        % Compute validation performance
                        testPerf = performanceMeasure( Ytest , YtestPred , testIdx );                

                        obj.testPerformance(obj.rfMapper.currentParIdx , filter.currentParIdx) = testPerf;
                    end

                    %%%%%%%%%%%%%%%%%%%%
                    % Store best model %
                    %%%%%%%%%%%%%%%%%%%%
                    
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        obj.mapParStar = obj.rfMapper.currentPar;
                        
                        %Update best filter parameter
                        obj.filterParStar = filter.currentPar;
                        
                        % Update best mapped samples
                        obj.XrfStar = obj.rfMapper.Xrf;
                        
                        % Update best projections matrix
                        obj.rfOmegaStar = obj.rfMapper.omega;
                        
                        % Update bestb coefficients
                        obj.rfBStar = obj.rfMapper.b;
                        
                        %Update best validation performance measurement
                        valM = valPerf;
                        
                        if ~recompute
                            
%                             % Update internal model samples matrix
%                             obj.Xmodel = Xtrain;
                            
                            % Update coefficients vector
                            obj.w = filter.weights;
                        end
                    end
                end
            end
            
            % Find best parameters from validation performance matrix
            
              %[row, col] = find(valPerformance <= min(min(valPerformance)));

%             obj.kerParStar = obj.kerParGuesses
%             obj.filterParStar = ...  
            
            % Print best kernel hyperparameter(s)
            display('Best feature map hyperparameter(s):')
            obj.mapParStar

            % Print best filter hyperparameter(s)
            display('Best filter hyperparameter(s):')
            obj.filterParStar
            
            % Best validation performance
            display('Best validation performance:')
            valM
                                     
            % Set best mapped samples
            obj.rfMapper.Xrf = obj.XrfStar;

            % Set best omega matrix
            obj.rfMapper.omega = obj.rfOmegaStar;
            
            % Set best b vector
%             obj.rfMapper.omega = obj.rfOmegaStar;
            
            % Set best mapping parameters
            obj.rfMapper.currentPar = obj.mapParStar;
                        
            if (nargin > 4) && (recompute)
                
                % Recompute kernel on the whole training set with the best
                % kernel parameter
                     
                C = obj.XrfStar' * obj.XrfStar;

%                 obj.rfMapper.init(Xtr, Xtr);
%                 kernel.compute(obj.kerParStar);

                % Recompute filter on the whole training set with the best
                % filter parameter
                
                % Normalization factors
                numSamples = size(obj.XrfStar , 1);
                
                filter.init( C  , obj.rfMapper.Xrf' * Ytr , numSamples);
                filter.compute(obj.filterParStar);
                
%                 % Update internal model samples matrix
%                 obj.Xmodel = Xtr;
                
                % Update coefficients vector
                obj.w = filter.weights;
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

