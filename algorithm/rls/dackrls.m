classdef dackrls < algorithm
    %DACKRLS Divide and Conquer Regularized Least Squares
    %   This algorithm implements Divide and Conquer Regularized Least Squares
    %   (Kernel Ridge Regression), proposed by Zhang Y., Duchi J. and
    %   Wainwright M. in "Divide and Conquer Kernel Ridge Regression",
    %   JMLR, 2013
    
    properties
        
        % I/O options
        storeFullTrainPerf  % Store full training performance matrix 1/0
        storeFullValPerf    % Store full validation performance matrix 1/0
        storeFullTestPerf   % Store full test performance matrix 1/0
        valPerformance      % Validation performance matrix
        trainPerformance    % Training performance matrix
        testPerformance     % Test performance matrix
        
        mGuesses        % Guesses vector of the number of chunks m for each partition
        numMGuesses     % Cardinality of number of splits guesses
        mStar           % Best number of splits
        mStarIdx        % Best number of splits index
        XtrSplit        % Contains the indexes of the samples of each split
        Xtr               % Training inputs
        NTr             % Total number of training samples
        nTr             % Number of training samples of the current partition
        partitionIdx    % Cell array containing the disjoint sample indexes sets of the current partitions
        trainIdx        % Training set indexes
        valIdx          % Validation set indexes
        
        c
        Xmodel
        
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
%             checkMap = @(x) isa(x,'featureMap');
            checkMap = @(x) isa(x,'function_handle');
            addRequired(p,'map',checkMap);

            % filter
%             checkFilter = @(x) isa(x,'filter');
            checkFilter = @(x) isa(x,'function_handle');
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
            checkMapParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'mapParGuesses',defaultMapParGuesses,checkMapParGuesses);                    
            
            % numMapParGuesses        % Number of map parameter guesses
            defaultNumMapParGuesses = [];
            checkNumMapParGuesses = @(x) isinteger(x) && x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses);        
            
            % filterParGuesses       % filter parameter guesses vector
            defaultFilterParGuesses = [];
            checkFilterParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'filterParGuesses',defaultFilterParGuesses,checkFilterParGuesses);                
            
            % numFilterParGuesses    % Number of filter parameter guesses vector
            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x) isinteger(x) && x > 0 ;            
            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses,checkNumFilterParGuesses);      
            
            % Parse function inputs
            parse(p, map, filter, mGuesses, varargin{:})
            
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
            end
               
            warning('DACKRLS does not support map and filter hyperparameter selection yet.');
        end
        
        function train(obj , Xtr , Ytr , performanceMeasure , recompute, validationPart, varargin)
            
                        
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
            
            obj.Xtr = p.Results.Xtr;
            
            Xte = p.Results.Xte;
            Yte = p.Results.Yte;

            % Training/validation sets splitting
%             shuffledIdx = randperm(size(Xtr,1));
            ntr = floor(size(Xtr,1)*(1-validationPart));
%             obj.trainIdx = shuffledIdx(1 : tmp1);
%             valIdx = shuffledIdx(tmp1 + 1 : end);
            obj.trainIdx = 1 : ntr;
            valIdx = ntr + 1 : size(Xtr,1);
            
            Xtrain = Xtr(obj.trainIdx,:);
            Ytrain = Ytr(obj.trainIdx,:);
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
                    obj.XtrSplit{i,j} = obj.trainIdx( (j-1) * chunkSize + 1 : j * chunkSize);
                end
            end
            
            %%% Model selection and training of numMGuesses ensambles of KRLS models (impera)
            display('Model selection and training of numMGuesses ensambles of KRLS models (impera)');
            
            % Full matrices for performance storage initialization
            if obj.storeFullTrainPerf == 1
                obj.trainPerformance = cell(obj.numMGuesses, obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            if obj.storeFullValPerf == 1
                obj.valPerformance = cell(obj.numMGuesses, obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            if obj.storeFullTestPerf == 1
                obj.testPerformance = cell(obj.numMGuesses, obj.numMapParGuesses, obj.numFilterParGuesses);
            end
            
            % Initializations
            obj.c = cell(obj.numMGuesses,max(obj.mGuesses),obj.numMapParGuesses,obj.numFilterParGuesses);
%             YvalPred = cell(obj.numMGuesses,max(obj.mGuesses),obj.numMapParGuesses,obj.numFilterParGuesses);
            valPerf = cell(1,obj.numMGuesses);
%             kernelVal = cell(obj.numMGuesses,max(obj.mGuesses),obj.numMapParGuesses,obj.numFilterParGuesses);
            valM = inf;     % Keeps track of the lowest validation error
                                
            for i = 1:obj.numMGuesses
                for j = 1:obj.mGuesses(i)
                    
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
                    kernelTrain = obj.map( Xtr(obj.XtrSplit{i,j},:) , Xtr(obj.XtrSplit{i,j},:) , argin{:});
                   
                    while kernelTrain.next()

                        % Compute kernels according to current hyperparameters
                        kernelTrain.compute();
                        
                        % Initialize regularization filter
                        argin = {};
                        if ~isempty(obj.numFilterParGuesses)
                            argin = [argin , 'numFilterParGuesses' , obj.numFilterParGuesses];
                        end
                        if ~isempty(obj.filterParGuesses)
                            argin = [argin , 'filterParGuesses' , obj.filterParGuesses];
                        end
                        if ~isempty(obj.verbose)
                            argin = [argin , 'verbose' , obj.verbose];
                        end
                        filter = obj.filter( kernelTrain.K , Ytr(obj.XtrSplit{i,j},:) , size(kernelTrain, 1) , argin{:});

                        while filter.next()

                            % Compute filter according to current hyperparameters
                            filter.compute();

                            % Update coefficients vector
                            obj.c{i,j} = filter.weights;
                            
                            obj.mapParStar = kernelTrain.currentPar;
                            obj.filterParStar = filter.currentPar;
                        end
                    end
                end
                
                %%%%%%%%%%%%%%%%%%%%%%
                % Combine estimators %
                %%%%%%%%%%%%%%%%%%%%%%

                % Compute partial predictions matrix
                partialPred = cell(1,obj.mGuesses(i));

                for j = 1:obj.mGuesses(i)

                    % Initialize TrainVal kernel
                    kernelVal = obj.map(Xval,Xtr(obj.XtrSplit{i,j},:));
                    kernelVal.compute(obj.mapParStar);

                    % Compute partial prediction
                    partialPred{j} = kernelVal.K * obj.c{i,j};
                end

                % Combine partial predictions
                fullPred = mean(cell2mat(partialPred),2);

                % Compute performance
                valPerf{i} = performanceMeasure( Yval , fullPred  , valIdx );

                % Populate full performance matrices
%                 trainPerformance(i,j) = performanceMeasure( kernel.K * filter.weights, Ytrain);
%                 obj.valPerformance = [obj.valPerformance valPerf];

                if valPerf{i} < valM
                    obj.mStar = obj.mGuesses(i);
                    obj.mStarIdx = i;

%                         % Update best kernel parameter combination
%                         obj.mapParStar = kernelTrain.currentPar;
% 
%                         %Update best filter parameter
%                         obj.filterParStar = filter.currentPar;

                    %Update best validation performance measurement
                    valM = valPerf{i};
                end                
            end
            
            % Print best kernel hyperparameter(s)
            display(['Best number of splits: ' , num2str(obj.mStar)]);
            
%             % Print best kernel hyperparameter(s)
%             display(['Best kernel hyperparameter(s): ' , num2str(obj.mapParStar)]);
% 
%             % Print best filter hyperparameter(s)
%             display(['Best filter hyperparameter(s): ' , num2str(obj.filterParStar)]);
%             
        end
        
        function Ypred = test( obj , Xte )

            % Compute partial predictions matrix
            partialPred = cell(1,obj.mStar);
            
            for j = 1:obj.mStar
                
                % Get kernel type and instantiate train-test kernel (including sigma)
                kernelTest = obj.map(Xte , obj.Xtr(obj.XtrSplit{obj.mStarIdx,j},:));
                kernelTest.compute(obj.mapParStar);

                % Compute partial prediction
                partialPred{j} = kernelTest.K * obj.c{obj.mStarIdx,j};
                
            end
            
            % Combine partial predictions into full prediction
            Ypred = mean(cell2mat(partialPred),2);
        end
    end
end

