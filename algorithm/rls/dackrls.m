classdef dackrls < algorithm
    %DACKRLS Divide and Conquer Regularized Least Squares
    %   This algorithm implements Divide and Conquer Regularized Least Squares
    %   (Kernel Ridge Regression), proposed by Zhang Y., Duchi J. and
    %   Wainwright M. in "Divide and Conquer Kernel Ridge Regression",
    %   JMLR, 2013
    
    properties
        
        % I/O options
        verbose             % 1: verbose; 0: silent        
        storeFullTrainPerf  % Store full training performance matrix 1/0
        storeFullValPerf    % Store full validation performance matrix 1/0
        storeFullTestPerf   % Store full test performance matrix 1/0
        valPerformance      % Validation performance matrix
        trainPerformance    % Training performance matrix
        testPerformance     % Test performance matrix
        
        
        mGuesses        % Guesses vector of the number of chunks m for each partition
        numMGuesses     
        X               % Full dataset inputs
        Y               % Full dataset outputs
        NTr             % Total number of training samples
        nTr             % Number of training samples of the current partition
        partitionIdx    % Cell array containing the disjoint sample indexes sets of the current partitions
        trainIdx        % Training set indexes
        valIdx          % Validation set indexes
        
        % Map properties (e.g. kernel or explicit feature map)
        map                     % Handle to the specified map
        mapParGuesses           % Map parameter guesses vector
        numMapParGuesses        % Number of map parameter guesses vector
%         numMapParRangeSamples   % Number of samples used for map parameter guesses range generation
        mapParStar              % Optimal selected map parameter

        % Filter properties
        filter                 % Handle to the specified filter
        filterParGuesses       % filter parameter guesses vector
        numFilterParGuesses    % Number of filter parameter guesses vector
        filterParStar          % Optimal selected filter parameter
        
    end
    
    methods
        
        function obj = dackrls(map, filter, mGuesses, varargin)
            init( obj , map, filter, mGuesses, varargin{:})
        end
        
        function init( obj , map, filter, mGuesses, varargin)

            p = inputParser;
            
            %%%% Required parameters
            
            % map
            checkMap = @(x) isa(x,'featureMap');
            addRequired(p,'map',checkMap);

            % filter
            checkFilter = @(x) isa(x,'filter');
            addRequired(p,'filter', checkFilter);
            
            % mGuesses
            checkMGuesses = @(x) size(x,2) > 0;
            addRequired(p,'mGuesses',checkMGuesses);
            
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
            checkMapParGuesses = @(x) iscell(x) && size(x,2) > 0 ;            
            addParameter(p,'mapParGuesses',defaultMapParGuesses,checkMapParGuesses);                    
            
            % numMapParGuesses        % Number of map parameter guesses
            defaultNumMapParGuesses = 10;
            checkNumMapParGuesses = @(x) isinteger(x) && x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses);        
            
            % filterParGuesses       % filter parameter guesses vector
            defaultFilterParGuesses = [];
            checkFilterParGuesses = @(x) iscell(x) && size(x,2) > 0 ;            
            addParameter(p,'filterParGuesses',defaultFilterParGuesses,checkFilterParGuesses);                
            
            % numFilterParGuesses    % Number of filter parameter guesses vector
            defaultNumFilterParGuesses = 10;
            checkNumFilterParGuesses = @(x) isinteger(x) && x > 0 ;            
            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses,checkNumFilterParGuesses);      
            
            addParameter(p,'verbose',defaultVerbose,checkVerbose)
            addParameter(p,'storeFullTrainPerf',defaultStoreFullTrainPerf,checkStoreFullTrainPerf)
            addParameter(p,'storeFullValPerf',defaultStoreFullValPerf,checkStoreFullValPerf)
            addParameter(p,'storeFullTestPerf',defaultStoreFullTestPerf,checkStoreFullTestPerf)
            addParameter(p,'mapParGuesses',defaultMapParGuesses,checkMapParGuesses)
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses, checkNumMapParGuesses)
            addParameter(p,'filterParGuesses',defaultFilterParGuesses,checkFilterParGuesses)
            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses, checkNumFilterParGuesses)
            
            % Parse function inputs
            parse(p, map, filter, mGuesses, varargin{:})
            
            % Assign parsed parameters to object properties
            props = properties(p.Results);
            for idx = 1:size(props,2)
                set(obj, props{idx} , p.Results.(props{idx}));
            end
            
            %%% Joint parameters validation
            
            if isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                error('either mapParGuesses or numMapParGuesses must be specified');
            end    
            
            if ~isempty(obj.mapParGuesses) && ~isempty(obj.numMapParGuesses)
                error('mapParGuesses and numMapParGuesses cannot be specified together');
            end
            
            if isempty(obj.filterParGuesses) && isempty(obj.numFilterParGuesses)
                error('either filterParGuesses or numFilterParGuesses must be specified');
            end         
            
            if ~isempty(obj.filterParGuesses) && ~isempty(obj.numFilterParGuesses)
                error('filterParGuesses and numFilterParGuesses cannot be specified together');
            end
               
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
            
            %%% Training set splitting in m disjoint chunks (divide)
            display('Training set splitting in m disjoint chunks (divide)');
            obj.numMGuesses = size(obj.mGuesses,2);
            obj.XtrSplit = cell(obj.numMGuesses,max(obj.mGuesses));
            
            for i = 1:obj.numMGuesses
                chunkSize = floor(ntr/obj.mGuesses(i));
                if chunkSize <=0
                    error('Invalid chunk size!');
                end
                for j = 1:obj.mGuesses(i)
                    obj.XtrSplitIdx{i,j} = trainIdx( (j-1) * chunkSize + 1 : j * chunkSize));
                end
            end
            
            %%% Model selection and training of numMGuesses ensambles of KRLS models (impera)
            display('Model selection and training of numMGuesses ensambles of KRLS models (impera)');
            
            % Full matrices for performance storage initialization
            if obj.storeFullTrainPerf == 1
                obj.trainPerformance = zeros(obj.numMGuesses, size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            if obj.storeFullValPerf == 1
                obj.valPerformance = zeros(obj.numMGuesses, size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
            if obj.storeFullTestPerf == 1
                obj.testPerformance = zeros(obj.numMGuesses, size(obj.mapParGuesses,2), size(obj.filterParGuesses,2));
            end
                    
            for i = 1:obj.numMGuesses
                for j = 1:obj.mGuesses(i)
                    
                    % Initialize TrainVal kernel
                    kernelVal = obj.map(Xval,Xtr(obj.XtrSplitIdx{i,j},:));

                    % Initialize Train kernel
                    argin = [];
                    if ~isempty(obj.numMapParGuesses)
                        argin = [argin , 'numMapParGuesses' , obj.numMapParGuesses];
                    end
                    if ~isempty(obj.mapParGuesses)
                        argin = [argin , 'mapParGuesses' , obj.mapParGuesses];
                    end
                    if ~isempty(obj.verbose)
                        argin = [argin , 'verbose' , obj.verbose];
                    end
                    kernelTrain = obj.map( Xtr(obj.XtrSplitIdx{i,j},:) , Xtr(obj.XtrSplitIdx{i,j},:) , argin);
                    
                    % Number of samples of the current disjoint training subset
                    numSamples = size(kernelTrain, 1);

                    valM = inf;     % Keeps track of the lowest validation error

                    while kernelTrain.next()

                        % Compute kernel according to current hyperparameters
                        kernelTrain.compute();
                        kernelVal.compute(kernel.currentPar);


                        % Initialize filter

                        argin = [];
                        if ~isempty(obj.numFilterParGuesses)
                            argin = [argin , 'numFilterParGuesses' , obj.numFilterParGuesses];
                        end
                        if ~isempty(obj.filterParGuesses)
                            argin = [argin , 'filterParGuesses' , obj.filterParGuesses];
                        end
                        if ~isempty(obj.verbose)
                            argin = [argin , 'verbose' , obj.verbose];
                        end

                        filter = obj.filter( kernelTrain.K , Ytr(obj.XtrSplitIdx{i,j},:) , numSamples , argin);

%                         obj.filterParGuesses = [obj.filterParGuesses ; filter.rng];

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
                
                end
            end
            
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
            
            
            
            % Find best parameters from validation performance matrix
            
              %[row, col] = find(valPerformance <= min(min(valPerformance)));

%             obj.kerParStar = obj.kerParGuesses
%             obj.filterParStar = ...  
            
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

