classdef sequentialResidualKrls < algorithm
    % sequentialResidualKrls This algorithm performs KRLS several times,
    % cross-validating sigma each time and learning on the output residual
    % of the previous step.
    
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
        Ktrain
        
        batchRank
        iterations
    end
    
    methods
        
        function obj = sequentialResidualKrls(map, filter, varargin)
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

            % Iterations
            defaultBatchRank = 1;            
            checkBatchRank = @(x) x > 0 ;
            addParameter(p,'batchRank',defaultBatchRank,checkBatchRank);      

            % Iterations
            defaultIterations = 1;            
            checkIterations = @(x) x > 0 ;
            addParameter(p,'iterations',defaultIterations,checkIterations);      
            
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
            
            % mapParStar       % Fixed kernel parameters array
            defaultMapParStar = [];
            checkMapParStar = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'mapParStar',defaultMapParStar,checkMapParStar);                    
            
            % numMapParGuesses        % Number of map parameter guesses
            defaultNumMapParGuesses = [];
            checkNumMapParGuesses = @(x) x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses);        
            
            % filterParGuesses       % filter parameter guesses vector
            defaultFilterParGuesses = [];
            checkFilterParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'filterParGuesses',defaultFilterParGuesses,checkFilterParGuesses);                
            
            % filterParStar       % Fixed filter parameters array
            defaultFilterParStar = [];
            checkFilterParStar = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'filterParStar',defaultFilterParStar,checkFilterParStar);                    
            
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
            
            if isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses) && isempty(obj.mapParStar)
                error('either mapParGuesses or numMapParGuesses or mapParStar must be specified');
            end    
            
            if ~isempty(obj.mapParGuesses) && ~isempty(obj.numMapParGuesses) || ...
               ~isempty(obj.mapParGuesses) && ~isempty(obj.mapParStar) || ...
               ~isempty(obj.mapParStar) && ~isempty(obj.numMapParGuesses)

                error('mapParGuesses, mapParStar and numMapParGuesses cannot be specified together');
            end    
            
            if ~isempty(obj.mapParStar) && (size(obj.mapParStar,2) ~= obj.iterations)
                error('The fixed kernel parameters must be specified for each iteration');
            end
            
            if ~isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                obj.numMapParGuesses = size(obj.mapParGuesses,2);
            end
            
            if isempty(obj.filterParGuesses) && isempty(obj.numFilterParGuesses) && isempty(obj.filterParStar)
                error('either filterParGuesses or numFilterParGuesses or filterParStar must be specified');
            end         
            

            if ~isempty(obj.filterParGuesses) && ~isempty(obj.numFilterParGuesses) || ...
               ~isempty(obj.filterParGuesses) && ~isempty(obj.filterParStar) || ...
               ~isempty(obj.filterParStar) && ~isempty(obj.numFilterParGuesses)

                error('filterParGuesses, filterParStar and numFilterParGuesses cannot be specified together');
            end    
            
            if ~isempty(obj.filterParStar) && (size(obj.filterParStar,2) ~= obj.iterations)
                error('The fixed filter parameters must be specified for each iteration');
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
            
            for iter = 1:obj.iterations

                % Initialize Train kernel
                argin = {};
                if ~isempty(obj.numMapParGuesses)
                    argin = [argin , 'numMapParGuesses' , obj.numMapParGuesses];
                end
                if ~isempty(obj.mapParGuesses)
                    argin = [argin , 'mapParGuesses' , obj.mapParGuesses];
                end
                if isempty(obj.mapParGuesses) && ~isempty(obj.mapParStar)
                    argin = [argin , 'mapParGuesses' , obj.mapParStar(iter)];
                end
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end
                kernelTrain = obj.map( Xtrain , Xtrain , argin{:});

%                 obj.mapParGuesses = kernelTrain.mapParGuesses;
%                 obj.filterParGuessesStorage = [];

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
                    argin = [argin , 'mapParGuesses' , full(kernelTrain.currentPar(1))];
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
                    if isempty(obj.filterParGuesses) && ~isempty(obj.filterParStar)
                        argin = [argin , 'filterParGuesses' , obj.filterParStar(iter)];
                    end
                    if ~isempty(obj.verbose)
                        argin = [argin , 'verbose' , obj.verbose];
                    end
                    filter = obj.filter( kernelTrain.K, Ytrain , numSamples , argin{:});

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

%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         %  Store performance matrices  %
%                         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         if obj.storeFullTrainPerf == 1         
% 
%                             % Compute training predictions matrix
%                             YtrainPred = kernelTrain.K * filter.weights;
% 
%                             % Compute training performance
%                             trainPerf = performanceMeasure( Ytrain , YtrainPred , trainIdx );
% 
%                             obj.trainPerformance(kernelTrain.currentParIdx , filter.currentParIdx) = trainPerf;
%                         end
% 
%                         if obj.storeFullValPerf == 1
%                             obj.valPerformance(kernelTrain.currentParIdx , filter.currentParIdx) = valPerf;
%                         end
%                         if obj.storeFullTestPerf == 1      
% 
%                             % Initialize TrainTest kernel
%                             argin = {};
%                             argin = [argin , 'mapParGuesses' , full(kernelTrain.currentPar)];
%                             if ~isempty(obj.verbose)
%                                 argin = [argin , 'verbose' , obj.verbose];
%                             end                  
%                             kernelTest = obj.map(Xte , Xtrain , argin{:});
%                             kernelTest.next();
%                             kernelTest.compute();
% 
%                             % Compute scores
%                             YtestPred = kernelTest.K * filter.weights;
% 
%                             % Compute training performance
%                             testPerf = performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );
% 
%                             obj.testPerformance(kernelTrain.currentParIdx , filter.currentParIdx) = testPerf;                        
%                         end

                        %%%%%%%%%%%%%%%%%%%%
                        % Store best model %
                        %%%%%%%%%%%%%%%%%%%%
                        if valPerf < valM

                            % Update best kernel parameter combination
                            if ~isempty(obj.mapParGuesses)
                                obj.mapParStar(iter) = kernelTrain.currentPar;
                            end
                            
                            %Update best filter parameter
                            if ~isempty(obj.filterParGuesses)
                                obj.filterParStar(iter) = filter.currentPar;
                            end

                            %Update best validation performance measurement
                            valM = valPerf;

                            if ~recompute

                                % Update internal model samples matrix
                                obj.Xmodel{iter} = Xtrain;

                                % Update coefficients vector
                                obj.c{iter} = filter.weights;
                                
                                % save kernel mat
                                obj.Ktrain{iter} = kernelTrain.K;
                            end
                        end
                    end
                end

                % Print best filter  and kernel hyperparameter(s)
                display(['Iteration #' , num2str(iter)])
                display(['Best kernel hyperparameter(s): ' , num2str(obj.mapParStar(iter))])
                display(['Best filter hyperparameter(s): ' , num2str(obj.filterParStar(iter))])
    
                % Compute residuals
                Ytrain = Ytrain - kernelTrain.K * obj.c{iter};
                Yval = Yval - kernelVal.K * obj.c{iter};
            end
        end
        
        function [Ypred,YpredSteps] = test( obj , Xte )
            
            YpredSteps  =cell(1,obj.iterations);
            Ypred = zeros(size(Xte,1),size(obj.c{1,1},2));
            
            for iter = 1:obj.iterations
                
                YpredSteps{iter} = zeros(size(Xte,1),size(obj.c{1,1},2));
                
                % Get kernel type and instantiate train-test kernel
                argin = {};
                argin = [argin , 'mapParGuesses' , full(obj.mapParStar(iter))];
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end
                kernelTest = obj.map(Xte , obj.Xmodel{iter} , argin{:});
                kernelTest.next();
                kernelTest.compute();

                YpredSteps{iter} = kernelTest.K * obj.c{iter};
                % Compute residuals
                Ypred = Ypred + YpredSteps{iter};
            end            
        end
    end
end

