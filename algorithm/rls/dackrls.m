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
        mapParStarIdx           % Optimal selected map parameter index

        % Filter properties
        filter                 % Handle to the specified filter
        filterParGuesses       % filter parameter guesses vector
        numFilterParGuesses    % Number of filter parameter guesses vector
        filterParStar          % Optimal selected filter parameter
        filterParStarIdx       % Optimal selected filter parameter index
        isFilterParGuessesFixed
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
            checkNumMapParGuesses = @(x) x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses);        
            
            % filterParGuesses       % filter parameter guesses vector
            defaultFilterParGuesses = [];
            checkFilterParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'filterParGuesses',defaultFilterParGuesses,checkFilterParGuesses);                
            
            % numFilterParGuesses    % Number of filter parameter guesses vector
            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x) x > 0 ;            
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
                obj.isFilterParGuessesFixed = 1;
            else
                obj.isFilterParGuessesFixed = 0;
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
            obj.valIdx = ntr + 1 : size(Xtr,1);
            
            Xtrain = Xtr(obj.trainIdx,:);
            Ytrain = Ytr(obj.trainIdx,:);
            Xval = Xtr(obj.valIdx,:);
            Yval = Ytr(obj.valIdx,:);
            
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
%                 obj.trainPerformance = cell(obj.numMGuesses, obj.numMapParGuesses, obj.numFilterParGuesses);
                obj.trainPerformance = cell(obj.numMGuesses,obj.numMapParGuesses,obj.numFilterParGuesses);
            end
            if obj.storeFullValPerf == 1
%                 obj.valPerformance = cell(obj.numMGuesses, obj.numMapParGuesses, obj.numFilterParGuesses);
                obj.valPerformance = cell(obj.numMGuesses,obj.numMapParGuesses,obj.numFilterParGuesses);
            end
            if obj.storeFullTestPerf == 1
%                 obj.testPerformance = cell(obj.numMGuesses, obj.numMapParGuesses, obj.numFilterParGuesses);
                obj.testPerformance = cell(obj.numMGuesses,obj.numMapParGuesses,obj.numFilterParGuesses);
            end
            
            % Initializations
            obj.c = cell(obj.numMGuesses,max(obj.mGuesses),obj.numMapParGuesses,obj.numFilterParGuesses);
            valPerf = cell(obj.numMGuesses,obj.numMapParGuesses,obj.numFilterParGuesses);
            valM = inf;     % Keeps track of the lowest validation error
                                
            for i = 1:obj.numMGuesses
                Ktr = cell(obj.mGuesses(i),obj.numMapParGuesses);
                for j = 1:obj.mGuesses(i)
                    
                    % Initialize Train kernel
                    argin = {};
                    if ~isempty(obj.numMapParGuesses)
                        argin = [argin , 'numMapParGuesses' , obj.numMapParGuesses];
                    end
                    if ~isempty(obj.mapParGuesses)
                        argin = [argin , 'mapParGuesses' , full(obj.mapParGuesses)];
                    end
                    if ~isempty(obj.verbose)
                        argin = [argin , 'verbose' , obj.verbose];
                    end
                    kernelTrain = obj.map( Xtr(obj.XtrSplit{i,j},:) , Xtr(obj.XtrSplit{i,j},:) , argin{:});
                   
                    while kernelTrain.next()

                        % Compute kernels according to current hyperparameters
                        kernelTrain.compute();
                        
                        if obj.storeFullTrainPerf == 1                    
                            Ktr{j,kernelTrain.currentParIdx} = kernelTrain.K;
                        end
                        
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

                            % Store coefficients vector
                            obj.c{i,j,kernelTrain.currentParIdx,filter.currentParIdx} = filter.weights;
                            
                        end
                    end
                end
                
                %%%%%%%%%%%%%%%%%%%%%%
                % Combine estimators %
                %%%%%%%%%%%%%%%%%%%%%%

                % Compute partial predictions matrix
                partialPred = cell(obj.mGuesses(i),obj.numMapParGuesses,obj.numFilterParGuesses);

                for k = 1:obj.numMapParGuesses
                    for l = 1:obj.numFilterParGuesses              
                        for j = 1:obj.mGuesses(i)

                            % Initialize TrainVal kernel
                            argin = {};
                            argin = [argin , 'mapParGuesses' , full(obj.mapParGuesses(k))];
                            if ~isempty(obj.verbose)
                                argin = [argin , 'verbose' , obj.verbose];
                            end                    
                            kernelVal = obj.map(Xval , Xtr(obj.XtrSplit{i,j},:) , argin{:});
                            kernelVal.next();
                            kernelVal.compute();

                            % Compute partial prediction
                            partialPred{j,k,l} = kernelVal.K * obj.c{i,j,k,l};
                        end

                        % Combine partial predictions
                        ctmp = partialPred(:,k,l);
                        fullPred = zeros(size(ctmp{1},1) , size(ctmp{1},2));
                        for ii = 1:numel(ctmp)
                            fullPred = fullPred + ctmp{ii};
                        end
                        fullPred = fullPred/numel(ctmp);

                        %%%%%%%%%%
                        % BUG!!! %
                        %%%%%%%%%%
                        
                        % Compute performance
                        valPerf{i,k,l} = performanceMeasure( Yval , fullPred );

                        if obj.storeFullValPerf == 1
                            obj.valPerformance{i,k,l} = valPerf{i,k,l};
                        end

                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Populate training perforance matrix %
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if obj.storeFullTrainPerf == 1                    

                            for j = 1:obj.mGuesses(i)

                                % Initialize TrainTrain kernel
                                argin = {};
                                argin = [argin , 'mapParGuesses' , full(obj.mapParStar)];
                                if ~isempty(obj.verbose)
                                    argin = [argin , 'verbose' , obj.verbose];
                                end                    
%                                 kernelTrain = obj.map(Xtr(obj.XtrSplit{i,j},:) , Xtr(obj.XtrSplit{i,j},:) , argin{:});
%                                 kernelTrain.next();
%                                 kernelTrain.compute();

                                % Compute partial prediction
                                partialPred{j,k,l} = Ktr{j,k} * obj.c{i,j,k,l};
                            end

                            % Combine partial predictions
                            ctmp = partialPred(:,k,l);
                            fullPred = zeros(size(ctmp{1},1) , size(ctmp{1},2));
                            for ii = 1:numel(ctmp)
                                fullPred = fullPred + ctmp{ii};
                            end
                            fullPred = fullPred/numel(ctmp);
                            
                            % Compute performance
                            obj.trainPerformance{i,k,l} = performanceMeasure( Ytr(obj.XtrSplit{i,j},:) , fullPred );
                        end

                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        %   Populate test perforance matrix   %
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        if obj.storeFullTestPerf == 1                    

                            for j = 1:obj.mGuesses(i)
                                % Initialize TrainTest kernel
                                argin = {};
                                argin = [argin , 'mapParGuesses' , full(obj.mapParGuesses(k))];
                                if ~isempty(obj.verbose)
                                    argin = [argin , 'verbose' , obj.verbose];
                                end                    
                                kernelTest = obj.map(Xte , Xtr(obj.XtrSplit{i,j},:) , argin{:});
                                kernelTest.next();
                                kernelTest.compute();

                                % Compute partial prediction
                                partialPred{j,k,l} = kernelTest.K * obj.c{i,j,k,l};
                            end

                            % Combine partial predictions
                            ctmp = partialPred(:,k,l);
                            fullPred = zeros(size(ctmp{1},1) , size(ctmp{1},2));
                            for ii = 1:numel(ctmp)
                                fullPred = fullPred + ctmp{ii};
                            end
                            fullPred = fullPred/numel(ctmp);
                            
                            % Compute performance
                            obj.testPerformance{i,k,l} = performanceMeasure( Yte , fullPred );
                        end

                        if valPerf{i,k,l} < valM
                            obj.mStar = obj.mGuesses(i);
                            obj.mStarIdx = i;

                            obj.mapParStar = obj.mapParGuesses(k);
                            obj.mapParStarIdx = k;

                            obj.filterParStar = obj.filterParGuesses(l);
                            obj.filterParStarIdx = l;

                            %Update best validation performance measurement
                            valM = valPerf{i,k,l};
                        end
                    end
                end
            end
            
            % Print best kernel hyperparameter(s)
            display(['Best number of splits: ' , num2str(obj.mStar)]);
            
            % Print best kernel hyperparameter(s)
            display(['Best kernel hyperparameter(s): ' , num2str(obj.mapParStar)]);

            % Print best filter hyperparameter(s)
            display(['Best filter hyperparameter(s): ' , num2str(obj.filterParStar)]);
        end
        
        function Ypred = test( obj , Xte )

            % Compute partial predictions matrix
            partialPred = cell(1,obj.mStar);
            
            for j = 1:obj.mStar
                
                % Get kernel type and instantiate train-test kernel (including sigma)
                argin = {};
                argin = [argin , 'mapParGuesses' , full(obj.mapParStar)];
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end
                kernelTest = obj.map(Xte , obj.Xtr(obj.XtrSplit{obj.mStarIdx,j},:) , argin{:});
                kernelTest.next();
                kernelTest.compute();

                % Compute partial prediction
%                 partialPred{j} = kernelTest.K * obj.c{obj.mStarIdx,j}; 
                partialPred{j} = kernelTest.K * obj.c{obj.mStarIdx,j,obj.mapParStarIdx,obj.filterParStarIdx}; 
                
            end
            
            % Combine partial predictions into full prediction
            ctmp = partialPred;
            Ypred = mean(reshape(cell2mat(ctmp), [ size(ctmp{1}), length(ctmp) ]), ndims(ctmp{1})+1);
        end
    end
end