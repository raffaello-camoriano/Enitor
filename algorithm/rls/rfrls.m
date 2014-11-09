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
        
        kerTy   % Approximated kernel type
        numKerParRangeSamples

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses
        filter
        
%         Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
        w       % Weights vector
    end
    
    methods
        
        function obj = rfrls(mapTy, kerTy , numKerParRangeSamples , filtTy,  numMapParGuesses , numFilterParGuesses)
            init( obj , mapTy, kerTy , numKerParRangeSamples , filtTy,  numMapParGuesses , numFilterParGuesses)
        end
        
        function init( obj , mapTy, kerTy , numKerParRangeSamples , filtTy,  numMapParGuesses , numFilterParGuesses)
            obj.mapType = mapTy;
            obj.kerTy = kerTy;
            obj.filterType = filtTy;
            obj.numKerParRangeSamples = numKerParRangeSamples;
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
                        
            % mapper instantiation
            obj.rfMapper = obj.mapType(Xtr , obj.numMapParGuesses , obj.numKerParRangeSamples , obj.kerTy);
            
            Ytrain = Ytr(trainIdx,:);
            Yval = Ytr(valIdx,:);                

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
                
                obj.filter = obj.filterType( 'primal' , C , Xtrain' * Ytrain , obj.numFilterParGuesses );
                
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
            
            if (nargin > 4) && (recompute)
                
                % Recompute kernel on the whole training set with the best
                % kernel parameter
                C = obj.rfMapper.Xrf' * obj.rfMapper.Xrf;

%                 obj.rfMapper.init(Xtr, Xtr);
%                 kernel.compute(obj.kerParStar);

                % Recompute filter on the whole training set with the best
                % filter parameter
                obj.filter.init('primal' , C , obj.rfMapper.Xrf' * Ytr);
                obj.filter.compute(obj.filterParStar);
                
%                 % Update internal model samples matrix
%                 obj.Xmodel = Xtr;
                
                % Update coefficients vector
                obj.w = obj.filter.weights;
            end        
        end
        
        function Ypred = test( obj , Xte )

            % Compute scores
            XteRF = obj.rfMapper.map(Xte);
            Ypred = XteRF * obj.w;

        end
    end
end

