classdef nystromUniform < nystrom
    %NYSTROMUNIFORM Implementation of a Nystrom approximation of the Gram
    %matrix.
    %
    % Input parameters:
    % X: n * d training matrix
    % numMapParGuesses: Number of guesses for the hyperparameters of the
    % mapper
    % numMapParRangeSamples: Number of samples used for estimating the
    % optimal range of the kernel hyperparameter
    % maxRank: Maximum rank of the Nystrom approximation
    
    properties
        prevPar
        
        numMapParRangeSamples   % Number of samples of X considered for estimating the maximum and minimum sigmas
        maxRank                 % Maximum rank of the kernel approximation
        
        numMapParGuesses
        mapParGuesses     % Fixed mapping parameter
        
        kernelType      % Type of approximated kernel
        sampledPoints   % Current sampled columns
        SqDistMat       % Squared distances matrix
        Xs              % Sampled points
        C               % Current n-by-l matrix, composed of the evaluations of the kernel function for the sampled columns
        W               % Current l-by-l matrix, s. t. K ~ C * W^{-1} * C^T
    end
    
    methods
        % Constructor
        function obj = nystromUniform( X , numNysParGuesses , numMapParGuesses , numMapParRangeSamples , maxRank , mapParGuesses , verbose)
            
            obj.init( X , numNysParGuesses , numMapParGuesses , numMapParRangeSamples , maxRank , mapParGuesses , verbose);
            
            warning('Kernel type set by default to "gaussian"');
            obj.kernelType = @gaussianKernel;
        end
        
        % Initialization function
        function obj = init(obj , X , numNysParGuesses , numMapParGuesses , numMapParRangeSamples , maxRank , mapParGuesses , verbose)
            
            obj.X = X;
            obj.numMapParRangeSamples = numMapParRangeSamples;
            obj.d = size(X , 2);     
            obj.maxRank = maxRank;
            obj.mapParGuesses = mapParGuesses;
            
%             if ~isempty(mapParGuesses)
%                 obj.numMapParGuesses = size(mapParGuesses;
%             else
                obj.numMapParGuesses = numMapParGuesses;
%             end
            
            obj.numNysParGuesses = numNysParGuesses;
            
            obj.verbose = 0;
            if verbose == 1
                obj.verbose = 1;
            end
            
            % Compute range
            obj.range();
            obj.currentParIdx = 0;
            obj.currentPar = [];
            obj.prevPar = [];
        end
        
        function obj = range(obj)
            %% Range of the number of sampled columns
            
            tmpl = round(linspace(1, obj.maxRank , obj.numNysParGuesses));   
            %warning('The rank of the approximated matrix is fixed to maxRank');
%             tmpl = obj.maxRank;
            
            %% Approximated kernel parameter range
            
            if isempty(obj.mapParGuesses)
                % Compute max and min sigma guesses

                % Extract an even number of samples without replacement                

                % WARNING: not compatible with versions older than 2014
                %samp = datasample( obj.X(:,:) , obj.numMapParRangeSamples - mod(obj.numMapParRangeSamples,2) , 'Replace', false);

                % WARNING: Alternative to datasample below
                nRows = size(obj.X,1); % number of rows
                nSample = obj.numMapParRangeSamples - mod(obj.numMapParRangeSamples,2); % number of samples
                rndIDX = randperm(nRows); 
                samp = obj.X(rndIDX(1:nSample), :);   

                % Compute squared distances  vector (D)
                numDistMeas = floor(obj.numMapParRangeSamples/2); % Number of distance measurements
                D = zeros(1 , numDistMeas);
                for i = 1:numDistMeas
                    D(i) = sum((samp(2*i-1,:) - samp(2*i,:)).^2);
                end
                D = sort(D);

%                 firstPercentile = round(0.01 * numel(D) + 0.5);
%                 minGuess = sqrt( D(firstPercentile));
%                 maxGuess = sqrt( max(D) );

                fifthPercentile = round(0.05 * numel(D) + 0.5);
                ninetyfifthPercentile = round(0.95 * numel(D) - 0.5);
                minGuess = sqrt( D(fifthPercentile));
                maxGuess = sqrt( D(ninetyfifthPercentile) );
                
                if minGuess <= 0
                    minGuess = eps;
                end
                if maxGuess <= 0
                    maxGuess = eps;
                end	

                tmpKerPar = linspace(minGuess, maxGuess , obj.numMapParGuesses);
                
            else
                tmpKerPar = obj.mapParGuesses;
            end

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
            
            obj.SqDistMat = repmat(Sx1 , 1 , size(X2,1)) -2*Sx1x2 + repmat(Sx2 , size(X1,1) , 1);
        
        end
        
        function compute(obj , mapPar)
            
            if( nargin > 1 )
                
                if(obj.verbose == 1)
                    disp('Mapping will be computed according to the provided hyperparameter(s)');
                    mapPar
                end
                chosenPar = mapPar;
                
            elseif (nargin == 1) && (isempty(obj.currentPar))
                
                % If any current value for any of the parameters is not available, abort.
                error('Mapping parameter(s) not explicitly specified, and some internal current parameters are not available available. Exiting...');
            else
                
                if(obj.verbose == 1)
                    disp('Mapping will be computed according to the current internal hyperparameter(s)');
                    obj.currentPar
                end
                chosenPar = obj.currentPar;
            end

            % Uniformly sample points (rows of X) with replacement
%             obj.sampledPoints = randi(size(obj.X,1),1,chosenPar(1));

                
            % Uniformly sample points (rows of X) without replacement
%             obj.sampledPoints = randperm(size(obj.X,1),chosenPar(1));
            % Check sample
            %scatter(1:size(obj.sampledPoints,2),obj.sampledPoints)
            
%             if ~isempty(obj.prevPar)
%                 newSampledPoints = obj.prevPar(1)+1:chosenPar(1);
%                 XsNew = obj.X(newSampledPoints ,:);
%                 obj.Xs = [ obj.Xs ; XsNew ];  
%                 obj.sampledPoints = [ obj.sampledPoints , newSampledPoints ];
%             else
                obj.sampledPoints = 1:chosenPar(1);
                obj.Xs = obj.X(obj.sampledPoints,:);
%             end
            
            % Compute C and W
            obj.computeSqDistMat(obj.X , obj.Xs);
            obj.C = exp(-obj.SqDistMat / (2 * chosenPar(2)^2));
            obj.SqDistMat = [];     % Erase square distance matrix
            obj.W = obj.C(obj.sampledPoints , :);
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
                obj.prevPar = obj.currentPar;
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.rng{obj.currentParIdx};
                available = true;
            end
        end        
    end
end
