classdef incrementalNkrls2 < algorithm
    % Incremental Nystrom KRLS with Tikhonov regularization filtering
    %   Detailed explanation goes here
    
    properties
        
        % I/O options
        storeFullTrainTime  % Store full training time matrix 1/0
        trainTime           % Training time matrix
        perfEvalStep		% Evaluate and store the performances every given steps. 1 by default
        
        ntr   % Number of training samples
        nval   % Number of validation samples
        nte   % Number of test samples
        
        % Validation and test kernels n*m. They grow with m incrementally
        KVal
        KTest
        
		% Kernel/mapper props
        nyMapper
        mapType
        numMapParRangeSamples
        mapParGuesses
        mapParStar
        mapParStarIdx
        numMapParGuesses
        minRank
        maxRank
        
        numNysParGuesses

        % Filter props
        filterType
        filterParStar
        filterParStarIdx
        filterParGuesses
        numFilterParGuesses    
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector        
        
        % Stopping rule
        stoppingRule        % Handle to the stopping rule
    end
    
    methods
        
        function o = incrementalNkrls2(mapType , maxRank , varargin)
            init( o , mapType, maxRank , varargin)
        end
        
        function init( o , mapType, maxRank , varargin)

            display('Note that incrementalNkrls uses the Tikhonov filter in this implementation.');
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
            
            % Initialize Nystrom Mapper
            argin = {};
            if ~isempty(o.numNysParGuesses)
                argin = [argin , 'numNysParGuesses' , o.numNysParGuesses];
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
            o.nyMapper = o.mapType(Xtrain, Ytrain , o.ntr , argin{:} );
            o.mapParGuesses = o.nyMapper.rng;   % Warning: rename to mapParGuesses
            
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
                
                o.nyMapper.resetPar();
                if ~isempty(o.stoppingRule)
                    o.stoppingRule.reset();
                end
                o.nyMapper.filterPar = o.filterParGuesses(i);
                
                o.KVal = zeros( o.nval , o.maxRank );
                o.KTest = zeros( o.nte , o.maxRank );
                
                while o.nyMapper.next()

                    if(o.verbose)
                        display(['nyMapper guess ' , num2str(o.nyMapper.currentParIdx) , ' of ' , num2str(size(o.nyMapper.rng,2))]);
                    end
                    
                    if o.storeFullTrainTime == 1
                        tic
                    end
                    
                    o.nyMapper.compute();
                    
                    if o.storeFullTrainTime == 1 && ((isempty(o.nyMapper.prevPar) && o.nyMapper.currentParIdx == 1) || ...
							(~isempty(o.nyMapper.prevPar) && o.nyMapper.currentPar(1) < o.nyMapper.prevPar(1)))
                        o.trainTime(o.nyMapper.currentParIdx , i) = toc;
                    elseif o.storeFullTrainTime == 1
                        o.trainTime(o.nyMapper.currentParIdx , i) = o.trainTime(o.nyMapper.currentParIdx - 1 , i) + toc;
                    end
                    
                    if (o.perfEvalStep == 1) || ...
                            ( mod(o.nyMapper.currentPar(1), o.perfEvalStep) == 0 )
                            
%                     	Update validation kernel KVal
						if o.nyMapper.currentPar(1) == 1
						
							argin = {};
							argin = [argin , 'mapParGuesses' , o.nyMapper.currentPar(2)];
							if ~isempty(o.verbose)
								argin = [argin , 'verbose' , o.verbose];
							end             
							kernelMaker = o.nyMapper.kernelType(Xval,o.nyMapper.Xs, argin{:});
							kernelMaker.next();
							kernelMaker.compute();									
							o.KVal(: , 1 : o.nyMapper.currentPar(1)) = kernelMaker.K;
							clear kernelMaker;
						
                        else
							argin = {};
							argin = [argin , 'mapParGuesses' , o.nyMapper.currentPar(2)];
							if ~isempty(o.verbose)
								argin = [argin , 'verbose' , o.verbose];
							end             
							kernelMaker = o.nyMapper.kernelType(Xval, ...
											o.nyMapper.Xs((o.nyMapper.currentPar(1) - o.perfEvalStep + 1) : o.nyMapper.currentPar(1),:), ...
											argin{:});
							kernelMaker.next();
							kernelMaker.compute();									
							o.KVal(: , (o.nyMapper.currentPar(1) - o.perfEvalStep + 1) : o.nyMapper.currentPar(1)) = ...
								kernelMaker.K;
							clear kernelMaker;
						
						end

						% Compute predictions matrix
						YvalPred = o.KVal(: , 1 : o.nyMapper.currentPar(1)) * o.nyMapper.alpha;

						% Compute validation performance
						valPerf = performanceMeasure( Yval , YvalPred , valIdx );                

						% Apply early stopping criterion
						stop = 0;
						if ~isempty(o.stoppingRule)
							stop = o.stoppingRule.evaluate(valPerf);
						end

						if stop == 1
							o.nyMapper.resetPar();
							break;
						end

						%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
						%  Store performance matrices  %
						%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

						if o.storeFullTrainPerf == 1                    

							% Compute training predictions matrix
							YtrainPred = ...
								o.nyMapper.A(:,1:o.nyMapper.currentPar(1))* o.nyMapper.alpha;

							% Compute training performance
							o.trainPerformance(o.nyMapper.currentParIdx , i) = ...
                                performanceMeasure( Ytrain , YtrainPred , trainIdx );
						end

						if o.storeFullValPerf == 1
							o.valPerformance(o.nyMapper.currentParIdx , i) = valPerf;
						end

						if o.storeFullTestPerf == 1                    

	%                     	Update test kernel KTest
							if o.nyMapper.currentPar(1) == 1
							
								argin = {};
								argin = [argin , 'mapParGuesses' , o.nyMapper.currentPar(2)];
								if ~isempty(o.verbose)
									argin = [argin , 'verbose' , o.verbose];
								end             
								kernelMaker = o.nyMapper.kernelType(Xte,o.nyMapper.Xs, argin{:});
								kernelMaker.next();
								kernelMaker.compute();									
								o.KTest(: , 1 : o.nyMapper.currentPar(1)) = kernelMaker.K;
                                clear kernelMaker;
							
                            else
								argin = {};
								argin = [argin , 'mapParGuesses' , o.nyMapper.currentPar(2)];
								if ~isempty(o.verbose)
									argin = [argin , 'verbose' , o.verbose];
								end             
								kernelMaker = o.nyMapper.kernelType(Xte, ...
												o.nyMapper.Xs((o.nyMapper.currentPar(1) - o.perfEvalStep + 1) : o.nyMapper.currentPar(1) , :), ...
												argin{:});
								kernelMaker.next();
								kernelMaker.compute();									
								o.KTest(: , (o.nyMapper.currentPar(1) - o.perfEvalStep + 1) : o.nyMapper.currentPar(1)) = ...
									kernelMaker.K;
                                clear kernelMaker;
							end

							% Compute scores
							YtestPred = o.KTest(: , 1 : o.nyMapper.currentPar(1)) * o.nyMapper.alpha;

							% Compute test performance
							o.testPerformance(o.nyMapper.currentParIdx , i) = ...
								performanceMeasure( Yte , YtestPred , 1:size(Yte,1) );
						end

						%%%%%%%%%%%%%%%%%%%%
						% Store best model %
						%%%%%%%%%%%%%%%%%%%%
						if valPerf < valM

							% Update best kernel parameter combination
							o.mapParStar = o.nyMapper.currentPar;
							
							% Update best kernel parameter combination index
							o.mapParStarIdx = o.nyMapper.currentParIdx;

							%Update best filter parameter
							o.filterParStar = o.nyMapper.filterPar;
							
							%Update best filter parameter index
							o.filterParStarIdx = i;

							%Update best validation performance measurement
							valM = valPerf;

							% Update internal model samples matrix
							o.Xmodel = o.nyMapper.Xs;

							% Update coefficients vector
							o.c = o.nyMapper.alpha;
						end
					end
                end
            end
            
            % Free memory
            if isfield(o.nyMapper,'R')
                o.nyMapper.R = [];
            end
            if isfield(o.nyMapper,'M')
                o.nyMapper.M = [];
            end
            o.nyMapper.alpha = [];
            o.nyMapper.Xs = [];
            
            if o.verbose == 1
                
                % Print best kernel hyperparameter(s)
                display('Best mapper hyperparameter(s):')
                o.mapParStar

                % Print best filter hyperparameter(s)
                display('Best filter hyperparameter(s):')
                o.filterParStar
            end
        end
        
        function justTrain(o , Xtr , Ytr)
                        
            p = inputParser;
            
            %%%% Required parameters
            
            addRequired(p,'Xtr');
            addRequired(p,'Ytr');   

            % Parse function inputs
            parse(p, Xtr , Ytr)
            
            o.ntr = size(Xtr,1);
            
            % Initialize Nystrom Mapper
            argin = {};
            if ~isempty(o.numNysParGuesses)
                argin = [argin , 'numNysParGuesses' , o.numNysParGuesses];
            end      
            if ~isempty(o.minRank)
                argin = [argin , 'minRank' , o.minRank];
            end      
            if ~isempty(o.maxRank)
                argin = [argin , 'maxRank' , o.maxRank];
            end      
            if ~isempty(o.mapParStar)
                argin = [argin , 'mapParGuesses' , full(o.mapParStar(2))];
            end      
%             if ~isempty(o.filterParGuesses)
%                 argin = [argin , 'filterParGuesses' , o.filterParGuesses];
%             end           
            if ~isempty(o.verbose)
                argin = [argin , 'verbose' , o.verbose];
            end
            o.nyMapper = o.mapType(Xtr, Ytr , o.ntr , argin{:} );
%             o.mapParGuesses = o.nyMapper.rng;   % Warning: rename to mapParGuesses
            
            o.nyMapper.filterParGuesses = o.filterParStar;    

            while o.nyMapper.next()
            
                o.nyMapper.compute();
           
            end

            %%%%%%%%%%%%%%%%%%%%%%%
            % Store trained model %
            %%%%%%%%%%%%%%%%%%%%%%%

            % Update internal model samples matrix
            o.Xmodel = o.nyMapper.Xs;

            % Update coefficients vector
            o.c = o.nyMapper.alpha{1};
            
            % Free memory
            o.nyMapper.M = [];
            o.nyMapper.alpha = [];
            o.nyMapper.Xs = [];
        end
        
        function Ypred = test( o , Xte )
                
%			if o.storeFullTestPerf == 1   
			                 
%				o.testPerformance(o.mapParStarIdx , o.filterParStarIdx);
%			else
				
				% Get kernel type and instantiate train-test kernel (including sigma)
				argin = {};
				argin = [argin , 'mapParGuesses' , o.mapParStar(2)];
				if ~isempty(o.verbose)
					argin = [argin , 'verbose' , o.verbose];
				end
				kernelTest = o.nyMapper.kernelType(Xte , o.Xmodel , argin{:});
				kernelTest.next();
				kernelTest.compute();
				
				% Compute scores
				Ypred = kernelTest.K * o.c;
				
%			end
        end        
    end
end

