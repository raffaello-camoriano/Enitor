classdef rfrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Feature mapping props
        mapType
        mapParGuesses
        mapParStar
        numMapParGuesses
        rfMapper
        XrfStar                 % Best mapping of the training set
        rfOmegaStar             % Best random omega matrix
        rfBStar                 % Best coefficients vector b
        numKerParRangeSamples   % Number of samples used for kernel hyperparameter range guesses creation
        maxNumRF                % Maximum number of random features to be used

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses
        filter
        
        w               % Weights vector
    end
    
    methods
        
        function obj = rfrls(mapTy , numKerParRangeSamples , filtTy,  numMapParGuesses , numFilterParGuesses , maxNumRF)
            init( obj , mapTy , numKerParRangeSamples , filtTy,  numMapParGuesses , numFilterParGuesses , maxNumRF)
        end
        
        function init( obj , mapTy , numKerParRangeSamples , filtTy,  numMapParGuesses , numFilterParGuesses , maxNumRF)
            obj.mapType = mapTy;
            obj.numKerParRangeSamples = numKerParRangeSamples;
            obj.filterType = filtTy;
            obj.numMapParGuesses = numMapParGuesses;
            obj.numFilterParGuesses = numFilterParGuesses;
            obj.maxNumRF = maxNumRF;
        end
        
        function train(obj , Xtr , Ytr , performanceMeasure , recompute, validationPart)
            
            % Training/validation sets splitting
            shuffledIdx = randperm(size(Xtr,1));
            tmp1 = floor(size(Xtr,1)*(1-validationPart));
            trainIdx = shuffledIdx(1 : tmp1);
            valIdx = shuffledIdx(tmp1 + 1 : end);

%             tmp1 = floor(size(Xtr,1)*(1-validationPart));
%             trainIdx = 1 : tmp1;
%             valIdx = tmp1 + 1 : size(Xtr,1);      
                
            Ytrain = Ytr(trainIdx,:);
            Yval = Ytr(valIdx,:);    
            
            % mapper instantiation
            obj.rfMapper = obj.mapType(Xtr , obj.numMapParGuesses , obj.numKerParRangeSamples , obj.maxNumRF);
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage
%             trainPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
%             valPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);

            while obj.rfMapper.next()
                
                % Map samples with new hyperparameters
                obj.rfMapper.compute();
                
                % Get mapped samples according to the new map parameters
                % combination
                Xtrain = obj.rfMapper.Xrf(trainIdx,:);
                Xval = obj.rfMapper.Xrf(valIdx,:);
                
                % Compute covariance matrix of the training samples
                C = Xtrain' * Xtrain;
                
                % Normalization factors
                numSamples = size(Xtrain , 1);
                
                obj.filter = obj.filterType( C  , Xtrain' * Ytrain , numSamples ,  obj.numFilterParGuesses);
                
                while obj.filter.next()
                    
                    % Compute filter according to current hyperparameters
                    obj.filter.compute();

                    % Populate full performance matrices
                    %trainPerformance(i,j) = perfm( kernel.K * obj.filter.coeffs, Ytrain);
                    %valPerformance(i,j) = perfm( kernelVal.K * obj.filter.coeffs, Yval);
                    
                    % Compute predictions matrix
                    YvalPred = Xval * obj.filter.weights;
                    
                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred );
                    
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        obj.mapParStar = obj.rfMapper.currentPar;
                        
                        %Update best filter parameter
                        obj.filterParStar = obj.filter.currentPar;
                        
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
                            obj.w = obj.filter.weights;
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
            obj.rfMapper.omega = obj.rfOmegaStar;
            
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
                
                obj.filter.init( C  , obj.rfMapper.Xrf' * Ytr , numSamples);
                obj.filter.compute(obj.filterParStar);
                
%                 % Update internal model samples matrix
%                 obj.Xmodel = Xtr;
                
                % Update coefficients vector
                obj.w = obj.filter.weights;
            end        
        end
        
        function Ypred = test( obj , Xte )
            
            % Set best omega
            obj.rfMapper.omega = obj.rfOmegaStar;
            
            % Set best b
            obj.rfMapper.b = obj.rfBStar;
            
            % Set best mapping parameters
            obj.rfMapper.currentPar = obj.mapParStar;
            
            % Compute scores
            XteRF = obj.rfMapper.map(Xte);
            Ypred = XteRF * obj.w;
        end
    end
end

