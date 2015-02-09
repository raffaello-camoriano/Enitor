classdef incrementalNkrls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        ntr   % Number of training samples
        
        % Kernel props
        mapType
        kernelType
        numKerParRangeSamples
        kerParGuesses
        mapParStar
        numMapParGuesses
        fixedMapPar
        maxRank
        
        numNysParGuesses

        % Filter props
        filterType
        filterParStar
        filterParGuesses
        numFilterParGuesses    
        fixedFilterPar
        
%         trainIdx    % Training indexes used internally in the actually performed training
%         valIdx      % Validation indexes used internally in the actually
%         performed validation
        
        Xmodel     % Training samples actually used for training. they are part of the learned model
        c       % Coefficients vector
    end
    
    methods
        
        function obj = incrementalNkrls(mapType, numKerParRangeSamples, numNysParGuesses,  numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose)
            init( obj , mapType, numKerParRangeSamples ,  numNysParGuesses, numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose)
        end
        
        function init( obj , mapType, numKerParRangeSamples ,  numNysParGuesses, numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose)
            obj.mapType = mapType;
            obj.numKerParRangeSamples = numKerParRangeSamples;
            obj.numNysParGuesses = numNysParGuesses;
            obj.numMapParGuesses = numMapParGuesses;
            obj.filterParGuesses = filterParGuesses;
            obj.maxRank = maxRank;
            obj.fixedMapPar = fixedMapPar;
            obj.verbose = verbose;
        end
        
        function train(obj , Xtr , Ytr , performanceMeasure , recompute, validationPart)
                        
            % Training/validation sets splitting
            shuffledIdx = randperm(size(Xtr,1));
            tmp1 = floor(size(Xtr,1)*(1-validationPart));
            trainIdx = shuffledIdx(1 : tmp1);
            valIdx = shuffledIdx(tmp1 + 1 : end);
            
            Xtrain = Xtr(trainIdx,:);
            Ytrain = Ytr(trainIdx,:);
            Xval = Xtr(valIdx,:);
            Yval = Ytr(valIdx,:);
            
            obj.ntr = size(Xtrain,1);
            
            % Train kernel
%             X , Y , numNysParGuesses , numMapParGuesses , filterParGuesses , numKerParRangeSamples , maxRank , fixedMapPar , verbose)
            
            nyMapper = obj.mapType(Xtrain, Ytrain , obj.ntr , obj.numNysParGuesses , obj.numMapParGuesses , obj.filterParGuesses , obj.numKerParRangeSamples , obj.maxRank , obj.fixedMapPar , obj.verbose);
            obj.kerParGuesses = nyMapper.rng;   % Warning: rename to mapParGuesses
%             obj.filterParGuesses = [];
            
            valM = inf;     % Keeps track of the lowest validation error
            
            % Full matrices for performance storage
%             trainPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);
%             valPerformance = zeros(obj.kerParGuesses, obj.filterParGuesses);

            YvalPred = zeros(size(Xval,1),size(obj.filterParGuesses,2));
            
            while nyMapper.next()
                
                nyMapper.compute();
                
                % Compute Kvm TrainVal kernel
                obj.kernelType = nyMapper.kernelType;
                kernelVal = obj.kernelType(Xval,nyMapper.Xs);
                kernelVal.compute(nyMapper.currentPar(2));
                
                for i = 1:size(obj.filterParGuesses,2)

                    % Compute predictions matrix
                    YvalPred(:,i) = kernelVal.K * nyMapper.alpha{i};

                    % Compute performance
                    valPerf = performanceMeasure( Yval , YvalPred(:,i) , valIdx );                

                    if valPerf < valM

                        % Update best kernel parameter combination
                        obj.mapParStar = nyMapper.currentPar;

                        %Update best filter parameter
                        obj.filterParStar = nyMapper.filterParGuesses(i);

                        %Update best validation performance measurement
                        valM = valPerf;

                        % Update internal model samples matrix
                        obj.Xmodel = nyMapper.Xs;

                        % Update coefficients vector
                        obj.c = nyMapper.alpha{i};
                    end
                end
            end
            
            % Find best parameters from validation performance matrix
            
              %[row, col] = find(valPerformance <= min(min(valPerformance)));

%             obj.kerParStar = obj.kerParGuesses
%             obj.filterParStar = ...  
            
            if obj.verbose == 1
                
                % Print best kernel hyperparameter(s)
                display('Best mapper hyperparameter(s):')
                obj.mapParStar

                % Print best filter hyperparameter(s)
                display('Best filter hyperparameter(s):')
                obj.filterParStar
                
            end
        end
        
        function Ypred = test( obj , Xte )

                                                    
%                 % Compute Kvm TrainVal kernel
%                 kernelVal = nyMapper.kernelType(Xval,nyMapper.Xs);
%                 kernelVal.compute(nyMapper.currentPar(2));
                
            % Get kernel type and instantiate train-test kernel (including sigma)
%             kernelTest = nyMapper.kernelType(Xte , obj.Xmodel);
            kernelTest = obj.kernelType(Xte , obj.Xmodel);
            kernelTest.compute(obj.mapParStar(2));
            
            % Compute scores
            Ypred = kernelTest.K * obj.c;
            
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % Numerically stable version
%             B = (kernelTest.K / (L') ) * sqrtDinv;
%             YvalPred = B * obj.c;
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end
end

