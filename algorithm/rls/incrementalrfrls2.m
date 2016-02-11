classdef incrementalrfrls2 < algorithm
    %incrementalrfrls Incremental random features with gaussian kernel
    %   At each iteration, adds new features
    
    properties
        
        % I/O options
        storeFullTrainTime  % Store full training time matrix 1/0
        trainTime           % Training time matrix
        perfEvalStep		% Evaluate and store the performances every given steps. 1 by default

        ntr   % Number of training samples
        nval   % Number of validation samples
        nte   % Number of test samples
        
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
        
        function o = incrementalrfrls2(mapType , maxRank , varargin)
            init( o , mapType, maxRank , varargin)
        end
        
        function init( o , mapType, maxRank , varargin)

            display('Note that incrementalrfrls uses the Tikhonov filter in this implementation.');
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
            
            % perfEvalStep		% Evaluate and store the performances every given steps. 1 by default
            defaultPerfEvalStep = 1;
            checkPerfEvalStep = @(x) (x > 0) || (x <= maxRank);            
            addParameter(p,'perfEvalStep',defaultPerfEvalStep,checkPerfEvalStep);
                
            
            % Parse function inputs
            if isempty(varargin{:})
                parse(p, mapType , maxRank )
            else
                parse(p, mapType , maxRank ,  varargin{:}{:})
            end
            
            % Assign parsed parameters to object properties
            fields = fieldnames(p.Results);
            for idx = 1:numel(fields)
                o.(fields{idx}) = p.Results.(fields{idx});
            end
            
            %%% Joint parameters validation
            
            if o.minRank > o.maxRank
                error('The specified minimum rank of the kernel approximation is larger than the maximum one.');
            end 
            
            if isempty(o.mapParGuesses) && isempty(o.numMapParGuesses)
                error('either mapParGuesses or numMapParGuesses must be specified');
            end    
            
            if ~isempty(o.mapParGuesses) && ~isempty(o.numMapParGuesses)
                error('mapParGuesses and numMapParGuesses cannot be specified together');
            end    
            
            if ~isempty(o.mapParGuesses) && isempty(o.numMapParGuesses)
                o.numMapParGuesses = size(o.mapParGuesses,2);
            end
            
            if isempty(o.filterParGuesses) && isempty(o.numFilterParGuesses)
                error('either filterParGuesses or numFilterParGuesses must be specified');
            end         
            
            if ~isempty(o.filterParGuesses) && ~isempty(o.numFilterParGuesses)
                error('filterParGuesses and numFilterParGuesses cannot be specified together');
            end
            
            if ~isempty(o.filterParGuesses) && isempty(o.numFilterParGuesses)
                o.numFilterParGuesses = size(o.filterParGuesses,2);
            end        
        end
        
        function train(o , Xtr , Ytr , performanceMeasure , recompute, validationPart , varargin)
                        
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
            tmp1 = floor(size(Xtr,1)*(1-validationPart));
            trainIdx = 1 : tmp1;
            valIdx = tmp1 + 1 : size(Xtr,1);
            
            Xtrain = Xtr(trainIdx,:);
            Ytrain = Ytr(trainIdx,:);
            Xval = Xtr(valIdx,:);
            Yval = Ytr(valIdx,:);
            
            o.ntr = size(Xtrain,1);
            o.nval = size(Xval,1);
            o.nte = size(Xte,1);
            
            % Initialize Random Features Mapper
            argin = {};
            if ~isempty(o.numRFParGuesses)
                argin = [argin , 'numRFParGuesses' , o.numRFParGuesses];
            end      
            if ~isempty(o.maxRank)
                argin = [argin , 'maxRank' , o.maxRank];
            end      
            if ~isempty(o.minRank)
                argin = [argin , 'minRank' , o.minRank];
            end      
            if ~isempty(o.numMapParGuesses)
                argin = [argin , 'numMapParGuesses' , o.numMapParGuesses];
            end      
            if ~isempty(o.mapParGuesses)
                argin = [argin , 'mapParGuesses' , full(o.mapParGuesses)];
            end
            if ~isempty(o.numMapParRangeSamples)
                argin = [argin , 'numMapParRangeSamples' , o.numMapParRangeSamples];
            end     
            if ~isempty(o.verbose)
                argin = [argin , 'verbose' , o.verbose];
            end
            o.rfMapper = o.mapType(Xtrain, Ytrain , o.ntr , argin{:} );
            o.mapParGuesses = o.rfMapper.rng;   % Warning: rename to mapParGuesses
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage initialization
            if o.storeFullTrainPerf == 1
                o.trainPerformance = NaN*zeros(size(o.mapParGuesses,2), size(o.filterParGuesses,2));
            end
            if o.storeFullValPerf == 1
                o.valPerformance = NaN*zeros(size(o.mapParGuesses,2), size(o.filterParGuesses,2));
            end
            if o.storeFullTestPerf == 1
                o.testPerformance = NaN*zeros(size(o.mapParGuesses,2), size(o.filterParGuesses,2));
            end
            if o.storeFullTrainTime == 1
                o.trainTime = NaN*zeros(size(o.mapParGuesses,2), size(o.filterParGuesses,2));
            end
            
            for i = 1:size(o.filterParGuesses,2)
                
                if(o.verbose)
                    display(['Filter guess ' , num2str(i) , ' of ' , num2str(size(o.filterParGuesses,2))]);
                end
                
				o.rfMapper.resetPar();
                if ~isempty(o.stoppingRule)
                    o.stoppingRule.reset();
                end
                o.rfMapper.filterPar = o.filterParGuesses(i);
                
                o.XValTilda = zeros( o.nval , o.maxRank );
                o.XTestTilda = zeros( o.nte , o.maxRank );
                
                while o.rfMapper.next()

                    if(o.verbose)
                        display(['rfMapper guess ' , num2str(o.rfMapper.currentParIdx) , ' of ' , num2str(size(o.rfMapper.rng,2))]);
                    end
                    
					if o.storeFullTrainTime == 1
                        tic
                    end
                    
                    o.rfMapper.compute();

                    if o.storeFullTrainTime == 1 && ((isempty(o.rfMapper.prevPar) && o.rfMapper.currentParIdx == 1) || ...
                            (~isempty(o.rfMapper.prevPar) && o.rfMapper.currentPar(1) < o.rfMapper.prevPar(1)))
                        o.trainTime(o.rfMapper.currentParIdx , i) = toc;
                    elseif o.storeFullTrainTime == 1
                        o.trainTime(o.rfMapper.currentParIdx , i) = o.trainTime(o.rfMapper.currentParIdx - 1 , i) + toc;
                    end
                    
                    if (o.perfEvalStep == 1) || ...
                            ( mod(o.rfMapper.currentPar(1), o.perfEvalStep) == 0 )
                            
%                     	Update the RF mappings of the validation points
						if o.rfMapper.currentPar(1) == 1
							
							o.XValTilda(: , 1 : o.rfMapper.currentPar(1)) = ...
								o.rfMapper.map(Xval, ...
									o.rfMapper.rng(1 , 1 : o.rfMapper.currentParIdx) );            
						
						else
										 
							o.XValTilda(: , (o.rfMapper.currentPar(1) - o.perfEvalStep + 1) : o.rfMapper.currentPar(1)) = ...
								o.rfMapper.map(Xval, (o.rfMapper.currentPar(1) - o.perfEvalStep + 1) : o.rfMapper.currentPar(1));
						end

						% Compute predictions matrix
						YvalPred = o.XValTilda(: , 1 : o.rfMapper.currentPar(1)) * o.rfMapper.alpha;

						% Compute validation performance
						valPerf = performanceMeasure( Yval , YvalPred , valIdx );                

						% Apply early stopping criterion
						stop = 0;
						if ~isempty(o.stoppingRule)
							stop = o.stoppingRule.evaluate(valPerf);
						end

						if stop == 1
							o.rfMapper.resetPar();
							break;
						end

						%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
						%  Store performance matrices  %
						%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

						if o.storeFullTrainPerf == 1                    

							% Compute training predictions matrix
							YtrainPred = ...
								o.rfMapper.A(:,1:o.rfMapper.currentPar(1)) * o.rfMapper.alpha;

							% Compute validation performance
							o.trainPerformance(o.rfMapper.currentParIdx , i) = performanceMeasure( Ytrain , YtrainPred , trainIdx );  
						end

						if o.storeFullValPerf == 1
							o.valPerformance(o.rfMapper.currentParIdx , i) = valPerf;
						end

						if o.storeFullTestPerf == 1                    

							% Update the RF mappings of the test points
							if o.rfMapper.currentPar(1) == 1
								
								o.XTestTilda(: , 1 : o.rfMapper.currentPar(1)) = o.rfMapper.map(Xte, ...
									o.rfMapper.rng(1 , 1 : o.rfMapper.currentPar(1)));                        
							else
								
								o.XTestTilda(: , (o.rfMapper.currentPar(1) - o.perfEvalStep + 1) : o.rfMapper.currentPar(1)) = ...
									o.rfMapper.map(Xte, (o.rfMapper.currentPar(1) - o.perfEvalStep + 1) : o.rfMapper.currentPar(1));
							end
							
	%                         Compute predictions matrix
							YtestPred = o.XTestTilda(: , 1 : o.rfMapper.currentPar(1)) * o.rfMapper.alpha;

							% Compute test performance
							o.testPerformance(o.rfMapper.currentParIdx , i) = ...
								performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );       
						end

						%%%%%%%%%%%%%%%%%%%%
						% Store best model %
						%%%%%%%%%%%%%%%%%%%%
						if valPerf < valM

							% Update best kernel parameter combination
							o.mapParStar = o.rfMapper.currentPar;

							% Update best filter parameter
							o.filterParStar = o.rfMapper.filterPar;

							% Update best validation performance measurement
							valM = valPerf;

							% Update coefficients vector
							o.w = o.rfMapper.alpha;
						
							% Update best mapped samples
							o.XrfStar = o.rfMapper.A;
							
							% Update best projections matrix
							o.rfOmegaStar = o.rfMapper.omega;
							
							% Update bestb coefficients
							o.rfBStar = o.rfMapper.b;
						end
					end
                end
            end
            
            % Free memory
            o.rfMapper.R = [];
            o.rfMapper.alpha = [];
            
            if o.verbose == 1
                
                % Print best kernel hyperparameter(s)
                display('Best mapper hyperparameter(s):')
                o.mapParStar

                % Print best filter hyperparameter(s)
                display('Best filter hyperparameter(s):')
                o.filterParStar
            end
        end
        
        function Ypred = test( o , Xte )
            
            % Set best omega
            o.rfMapper.omega = o.rfOmegaStar;
            
            % Set best b
            o.rfMapper.b = o.rfBStar;
            
            % Set best mapping parameters
            o.rfMapper.currentPar = o.mapParStar;
            
            % Map test data
            XteRF = o.rfMapper.map(Xte,[]);

            % Compute predictions matrix
            Ypred = XteRF * o.w;
        end        
    end
end

