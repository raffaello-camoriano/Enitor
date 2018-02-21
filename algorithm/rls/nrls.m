classdef nrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        ntr
        
        storeFullTrainTime  % Store full training time matrix 1/0
        trainTime           % Training time matrix
        
        % Kernel props
        nyMapper
        mapType
        numMapParRangeSamples
        mapParGuesses
        numMapParGuesses
        mapParStar
        
        % Nystrom props
        minRank
        maxRank
        numNysParGuesses

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses    
        
        Xmodel      % Training samples actually used for training. they are part of the learned model
        c           % Coefficients vector
    end
    
    methods
        
        function obj = nrls( mapType, filterType , maxRank , varargin)
            init( obj , mapType, filterType , maxRank , varargin)
        end
        
        function init( obj ,  mapType, filterType , maxRank , varargin)


            p = inputParser;
            
            %%%% Required parameters
            
            checkMaxRank = @(x) x > 0 ;

            addRequired(p,'mapType');
            addRequired(p,'filterType');
            addRequired(p,'maxRank',checkMaxRank);
            
            %%%% Optional parameters
            % Optional parameter names:

            defaultMinRank = 1;            
            checkMinRank = @(x) x > 0 ;
            addParameter(p,'minRank',defaultMinRank,checkMinRank);                    
            
            defaultNumNysParGuesses = 1;            
            checkNumNysParGuesses = @(x) x > 0 ;
            addParameter(p,'numNysParGuesses',defaultNumNysParGuesses,checkNumNysParGuesses);                    
            
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
            
            % filterParGuesses       % Map parameter guesses cell array
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
            
            % Parse function inputs
            if isempty(varargin{:})
                parse(p, mapType , filterType , maxRank )
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

            % Train kernel
            obj.nyMapper = obj.mapType(Xtrain, obj.numNysParGuesses , obj.numMapParGuesses , obj.numMapParRangeSamples , obj.minRank, obj.maxRank , obj.mapParGuesses , obj.verbose);
            obj.mapParGuesses = obj.nyMapper.rng;   % Warning: rename to mapParGuesses
%             obj.filterParGuesses = [];
            
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

            while obj.nyMapper.next()
                
                if(obj.verbose)
                    display(['nyMapper guess ' , num2str(obj.nyMapper.currentParIdx) , ' of ' , num2str(size(obj.nyMapper.rng,2))]);
                end

                if obj.storeFullTrainTime == 1
                    tic
                end

                % Compute kernel according to current hyperparameters
                obj.nyMapper.compute();

                % Initialize TrainVal kernel
                argin = {};
                argin = [argin , 'mapParGuesses' , obj.nyMapper.currentPar(2)];
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end                    
                kernelVal = obj.nyMapper.kernelType(Xval,obj.nyMapper.Xs, argin{:});
                kernelVal.next();
                kernelVal.compute();                
                          
                if ~isempty(obj.filterParGuesses)
                    filter = obj.filterType( obj.nyMapper.C' * obj.nyMapper.C, obj.nyMapper.C' * Ytrain, obj.ntr , 'numFilterParGuesses' , obj.numFilterParGuesses, 'M' , obj.nyMapper.W , 'filterParGuesses' , obj.filterParGuesses , 'verbose' , obj.verbose);
                else
                    filter = obj.filterType( obj.nyMapper.C' * obj.nyMapper.C, obj.nyMapper.C' * Ytrain, obj.ntr , 'numFilterParGuesses' , obj.numFilterParGuesses, 'M' , obj.nyMapper.W , 'verbose' , obj.verbose);
                end
                
                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    if obj.storeFullTrainTime == 1 && ((isempty(obj.nyMapper.prevPar) && obj.nyMapper.currentParIdx == 1) || (~isempty(obj.nyMapper.prevPar) && obj.nyMapper.currentPar(1) < obj.nyMapper.prevPar(1)))
                        obj.trainTime(obj.nyMapper.currentParIdx , filter.currentParIdx) = toc;
                    elseif obj.storeFullTrainTime == 1
                        obj.trainTime(obj.nyMapper.currentParIdx , filter.currentParIdx) = obj.trainTime(obj.nyMapper.currentParIdx - 1 , filter.currentParIdx) + toc;
                    end

                    % Compute predictions matrix
                    YvalPred = kernelVal.K * filter.weights;

                    % Compute performance
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
                        YtrainPred = kernelTrain.K * filter.weights;
                        
                        % Compute training performance
                        trainPerf = performanceMeasure( Ytrain , YtrainPred , trainIdx );
                        
                        obj.trainPerformance(obj.nyMapper.currentParIdx , filter.currentParIdx) = trainPerf;
                    end
                    
                    if obj.storeFullValPerf == 1
                        obj.valPerformance(obj.nyMapper.currentParIdx , filter.currentParIdx) = valPerf;
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
                        YtestPred = kernelTest.K * filter.weights;

                        % Compute training performance
                        testPerf = performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );
                        
                        obj.testPerformance(obj.nyMapper.currentParIdx , filter.currentParIdx) = testPerf;                        
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%
                    % Store best model %
                    %%%%%%%%%%%%%%%%%%%%
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        obj.mapParStar = obj.nyMapper.currentPar;
                        
                        %Update best filter parameter
                        obj.filterParStar = filter.currentPar;
                        
                        %Update best validation performance measurement
                        valM = valPerf;
                        
                        % Update internal model samples matrix
                        obj.Xmodel = obj.nyMapper.Xs;

                        % Update coefficients vector
                        obj.c = filter.weights;
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

        function justTrain(obj , Xtr , Ytr)
                        
            p = inputParser;
            
            %%%% Required parameters
            
            addRequired(p,'Xtr');
            addRequired(p,'Ytr');   

            % Parse function inputs
            parse(p, Xtr , Ytr)
            
            obj.ntr = size(Xtr,1);
            
            % Initialize Nystrom Mapper
            argin = {};
            if ~isempty(obj.numNysParGuesses)
                argin = [argin , 'numNysParGuesses' , 1];
            end      
            if ~isempty(obj.maxRank)
                argin = [argin , 'maxRank' , obj.mapParStar(1)];
            end    
            if ~isempty(obj.minRank)
                argin = [argin , 'minRank' , obj.minRank];
            end        
            if ~isempty(obj.mapParStar)
                argin = [argin , 'mapParGuesses' , full(obj.mapParStar(2))];
            end      
%             if ~isempty(obj.filterParGuesses)
%                 argin = [argin , 'filterParGuesses' , obj.filterParGuesses];
%             end           
            if ~isempty(obj.verbose)
                argin = [argin , 'verbose' , obj.verbose];
            end
            
            obj.nyMapper = obj.mapType(Xtr, 1 , [] ,[] , [], obj.mapParStar(1) , full(obj.mapParStar(2)) , obj.verbose);
            obj.nyMapper.next();
            obj.nyMapper.compute();
            
            if ~isempty(obj.filterParGuesses)
                filter = obj.filterType( obj.nyMapper.C' * obj.nyMapper.C, obj.nyMapper.C' * Ytr, obj.ntr , 'numFilterParGuesses' , obj.numFilterParGuesses, 'M' , obj.nyMapper.W , 'filterParGuesses' , obj.filterParGuesses , 'verbose' , obj.verbose);
            else
                filter = obj.filterType( obj.nyMapper.C' * obj.nyMapper.C, obj.nyMapper.C' * Ytr, obj.ntr , 'numFilterParGuesses' , obj.numFilterParGuesses, 'M' , obj.nyMapper.W , 'verbose' , obj.verbose);
            end

%                 if ~isempty(obj.filterParGuesses)
%                     filter = obj.filterType( obj.nyMapper.C' * obj.nyMapper.C, obj.nyMapper.C' * Ytrain, obj.ntr , 'numFilterParGuesses' , obj.numFilterParGuesses, 'M' , eye(size(obj.nyMapper.W,1)) , 'filterParGuesses' , obj.filterParGuesses , 'verbose' , obj.verbose);
%                 else
%                     filter = obj.filterType( obj.nyMapper.C' * obj.nyMapper.C, obj.nyMapper.C' * Ytrain, obj.ntr , 'numFilterParGuesses' , obj.numFilterParGuesses, 'M' , eye(size(obj.nyMapper.W,1)) , 'verbose' , obj.verbose);
%                 end

            filter.next();
            
            % Compute filter according to current hyperparameters
            filter.compute();
            
            %%%%%%%%%%%%%%%%%%%%%%%
            % Store trained model %
            %%%%%%%%%%%%%%%%%%%%%%%
                        
            % Update internal model samples matrix
            obj.Xmodel = obj.nyMapper.Xs;

            % Update coefficients vector
            obj.c = filter.weights;
            
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

