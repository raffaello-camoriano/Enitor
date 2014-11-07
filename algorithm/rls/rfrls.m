classdef rfrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        % Feature mapping props
        mapType
        mapParGuesses
        mapParStar
        numMapParGuesses

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses        
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
        w       % Weights vector
    end
    
    methods
        
        function obj = rfrls(mapTy, filtTy,  numMapParGuesses , numFilterParGuesses)
            init( obj , mapTy, filtTy,  numMapParGuesses , numFilterParGuesses)
        end
        
        function init( obj , mapTy, filtTy,  numMapParGuesses , numFilterParGuesses)
            obj.mapType = mapTy;
            obj.filterType = filtTy;
            obj.numMapParGuesses = numMapParGuesses;
            obj.numFilterParGuesses = numFilterParGuesses;
        end
        
        function lambda = computeSingleLambda( lambdas)
            lambda = median(lambdas);
        end
        
        function train(obj , Xtr , Ytr , performanceMeasure , recompute, validationPart)
            
            % Training/validation sets splitting
            shuffledIdx = randperm(size(Xtr,1));
            tmp1 = floor(size(Xtr,1)*(1-validationPart));
            trainIdx = shuffledIdx(1 : tmp1);
            valIdx = shuffledIdx(tmp1 + 1 : end);
                        
            % mapping instantiation
            map = obj.mapType(Xtr);
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage
%             trainPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
%             valPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);

            while map.next()
                
                Xtrain = map.XtrRF(trainIdx,:);
                Ytrain = map.YtrRF(trainIdx,:);
                C = Xtrain' * Xtrain;   % Covariance matrix of the training samples
                Xval = map.XtrRF(valIdx,:);
                Yval = map.YtrRF(valIdx,:);                
                
                filter = obj.filterType( C , Ytrain , obj.numFilterParGuesses);
                
                while filter.next()
                    
                    % Compute filter according to current hyperparameters
                    filter.compute();

                    % Populate full performance matrices
                    %trainPerformance(i,j) = perfm( kernel.K * filter.coeffs, Ytrain);
                    %valPerformance(i,j) = perfm( kernelVal.K * filter.coeffs, Yval);
                    
                    % Compute predictions matrix
                    YvalPred = obj.test( Xval );
                    
                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred );
                    
                    if valPerf < valM
                        
                        % Update best kernel parameter combination
                        obj.mapParStar = map.currentPar;
                        
                        %Update best filter parameter
                        obj.filterParStar = filter.currentPar;
                        
                        %Update best validation performance measurement
                        valM = valPerf;
                        
                        if ~recompute
                            
                            % Update internal model samples matrix
                            obj.Xmodel = Xtrain;
                            
                            % Update coefficients vector
                            obj.c = filter.coeffs;
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
            
            if (nargin > 4) && (recompute)
                
                % Recompute kernel on the whole training set with the best
                % kernel parameter
                C = map.XtrRF' * map.XtrRF;
%                 map.init(Xtr, Xtr);
%                 kernel.compute(obj.kerParStar);

                % Recompute filter on the whole training set with the best
                % filter parameter
                filter.init(C,Ytr);
                filter.compute(obj.filterParStar);
                
                % Update internal model samples matrix
                obj.Xmodel = Xtr;
                
                % Update coefficients vector
                obj.c = filter.coeffs;
                obj.w = filter.weights; ???
            end        
        end
        
        function Ypred = test( obj , Xte )

            % Compute scores
            Ypred = Xte * obj.w;

        end
    end
end

