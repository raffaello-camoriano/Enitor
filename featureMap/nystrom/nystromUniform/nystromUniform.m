classdef nystromUniform < nystrom
    %RANDOMFEATURES Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numKerParRangeSamples   % Number of samples of X considered for estimating the maximum and minimum sigmas
        maxRank                 % Maximum rank of the kernel approximation
        
        kernelType      % Type of approximated kernel
        sampledPoints   % Current sampled columns
        SqDistMat       % Squared distances matrix
        Xs              % Sampled points
        C               % Current n-by-l matrix, composed of the evaluations of the kernel function for the sampled columns
        W               % Current l-by-l matrix, s. t. K ~ C * W^{-1} * C^T
    end
    
    methods
        % Constructor
        function obj = nystromUniform( X , numMapParGuesses , numKerParRangeSamples , maxRank )
            
            obj.init( X , numMapParGuesses , numKerParRangeSamples , maxRank );
            
            obj.kernelType = @gaussianKernel;
        end
        
        % Initialization function
        function obj = init(obj , X , numMapParGuesses , numKerParRangeSamples , maxRank)
            
            obj.X = X;
            obj.numMapParGuesses = numMapParGuesses;
            obj.numKerParRangeSamples = numKerParRangeSamples;
            obj.d = size(X , 2);     
            obj.maxRank = maxRank;
            
            % Compute range
            obj.range();
            obj.currentParIdx = 0;
            obj.currentPar = [];
        end
        
        function obj = range(obj)
            %% Range of the number of sampled columns
            
            tmpl = round(linspace(obj.maxRank/10, obj.maxRank , obj.numMapParGuesses));   
            %tmpl = obj.maxRank;
            
            %% Approximated kernel parameter range
            
            % Compute max and min sigma guesses
                
            % Extract an even number of samples without replacement                
            
            % WARNING: not compatible with versions older than 2014
            %samp = datasample( obj.X(:,:) , obj.numKerParRangeSamples - mod(obj.numKerParRangeSamples,2) , 'Replace', false);
            
            % WARNING: Alternative to datasample below
            nRows = size(obj.X,1); % number of rows
            nSample = obj.numKerParRangeSamples - mod(obj.numKerParRangeSamples,2); % number of samples
            rndIDX = randperm(nRows); 
            samp = obj.X(rndIDX(1:nSample), :);   
            
            % Compute squared distances  vector (D)
            D = zeros(1,obj.numKerParRangeSamples);
            for i = 1:2:obj.numKerParRangeSamples

                D(i) = sum((samp(i,:) - samp(i+1,:)).^2);
            end
            D = sort(D);

            firstPercentile = round(0.01 * numel(D) + 0.5);
            minGuess = sqrt( D(firstPercentile));
            maxGuess = sqrt( max(D) );

            if minGuess <= 0
                minGuess = eps;
            end
            if maxGuess <= 0
                maxGuess = eps;
            end	
            
            tmpKerPar = linspace(minGuess, maxGuess , obj.numMapParGuesses);
            
            %% Generate all possible parameters combinations            
            
            [p,q] = meshgrid(tmpl, tmpKerPar);
            tmp = [p(:) q(:)]';
            
            obj.rng = num2cell(tmp , 1);
            
        end
        
        % Computes the squared distance matrix SqDistMat based on X1, X2
        function computeSqDistMat(obj , X1 , X2)
            
            Sx1 = sum( X1.*X1 , 2);
            Sx2 = sum( X2.*X2 , 2)';
            Sx1x2 = X1 * X2';
            
            obj.SqDistMat = repmat(Sx1 , 1 , obj.m) -2*Sx1x2 + repmat(Sx2 , obj.n , 1);
        
        end
        
        function compute(obj , mapPar)
            
            if( nargin > 1 )
                
                disp('Mapping will be computed according to the provided hyperparameter(s)');
                mapPar
                chosenPar = mapPar;
                
            elseif (nargin == 1) && (isempty(obj.currentPar))
                
                % If any current value for any of the parameters is not available, abort.
                error('Mapping parameter(s) not explicitly specified, and some internal current parameters are not available available. Exiting...');
            
            else
                
                disp('Mapping will be computed according to the current internal hyperparameter(s)');
                obj.currentPar
                chosenPar = obj.currentPar;
                
            end

            % Uniformly sample points
            
            obj.sampledPoints = randi(size(obj.X,1),1,chosenPar(1));
            obj.Xs = obj.X(obj.sampledPoints,:);
            
            % Compute C and W
            obj.computeSqDistMat(X , obj.Xs);
            obj.C = exp(-obj.SqDistMat / (2 * obj.chosenPar(2)^2));
            obj.SqDistMat = [];     % Erase square distance matrix
            obj.W = obj.C(obj.sampledCols , :);
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            % If any range for any of the parameters is not available, recompute all ranges.
            if cellfun(@isempty , obj.rng)
                obj.range();
            end

            available = false;
            if length(obj.rng) > obj.currentParIdx
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.rng{obj.currentParIdx};
                available = true;
            end
        end        
    end
end
