classdef nystromLRLS < nystrom
    % NYSTROMLRLS Nystrom sampling with lambda-ridge leverage scores
    %   This class implements the approximation of the kernel matrix
    %   according to the Nystrom sampling method with non-uniform sampling
    %   probabilities depending on the approximate LRLS of each data point. It
    %   estimates the LRLS of each data point according to the algorithm reported in El Alaoui A. et
    %   Mahoney M. W., "Fast Randomized Kernel Methods With Statistical
    %   Guarantees", 2014.
    %
    % Input parameters:
    % X: n * d training matrix
    % numMapParGuesses: Number of guesses for the hyperparameters of the
    % mapper
    % numKerParRangeSamples: Number of samples used for estimating the
    % optimal range of the kernel hyperparameter
    % maxRank: Maximum rank of the Nystrom approximation
    
    properties
        numKerParRangeSamples   % Number of samples of X considered for estimating the maximum and minimum sigmas
        maxRank                 % Maximum rank of the kernel approximation
        
        approxLrls      % Approximate lambda-ridge leverage scores
        P               % Data points sampling probability array
        kernelType      % Type of approximated kernel
        sampledPoints   % Current sampled columns
        SqDistMat       % Squared distances matrix
        Xs              % Sampled points
        C               % Current n-by-l matrix, composed of the evaluations of the kernel function for the sampled columns
        W               % Current l-by-l matrix, s. t. K ~ C * W^{-1} * C^T
    end
    
    methods
        % Constructor
        function obj = nystromLRLS( X , numMapParGuesses , numKerParRangeSamples , maxRank )
            
            obj.init( X , numMapParGuesses , numKerParRangeSamples , maxRank );
            
            warning('Kernel type set by default to "gaussian"');
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
            
            %tmpl = round(linspace(obj.maxRank/10, obj.maxRank , obj.numMapParGuesses));   
            %warning('The rank of the approximated matrix is fixed to maxRank');
            tmpl = obj.maxRank;
            
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
            numDistMeas = floor(obj.numKerParRangeSamples/2); % Number of distance measurements
            D = zeros(1 , numDistMeas);
            for i = 1:numDistMeas
                D(i) = sum((samp(2*i-1,:) - samp(2*i,:)).^2);
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
            
            obj.SqDistMat = repmat(Sx1 , 1 , size(X2,1)) -2*Sx1x2 + repmat(Sx2 , size(X1,1) , 1);
        
        end
        
        % Computes the approximated Gram matrix, using the non-uniformly
        % sampled data points. Thus, it also computes the approximated
        % LRLSs first.
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

            %%% Compute the approximated LRLSs
            
            sampledPoints = randi(size(obj.X,1),1,chosenPar(1));
            Xs = obj.X(sampledPoints,:);
            
            % Compute C and W
            obj.computeSqDistMat(obj.X , Xs);
            C = exp(-obj.SqDistMat / (2 * chosenPar(2)^2));
            obj.SqDistMat = [];     % Erase square distance matrix
            W = C(sampledPoints , :);
            
            % Decompose W (LDL decomposition)
            % Add 10e-07 to W's diag before decomposing

            [L,D] = ldl(W+10^(-7)*eye(size(W,1)));
            %R = chol(W , 'upper');
            
            % Construct B (nXp) s. t. BB'=CW^{+}C'
            %B = R\C;
            
            B = C * pinv(L) * inv(sqrtm(D));
            
            % Compute approx scores
            obj.approxLrls = zeros(1, size(obj.X,1));
            lambda = 0.1;
            warning('lambda set to 0.1 in LRLS approximation. It shall be optimized.');
%             for i = 1:size(obj.X,1)
%                obj.approxLrls(i) = B(i,:) * (B'*B + size(obj.X,1) * lambda * eye(size(B,2)) ) * B(i,:)'; 
%             end
%             obj.approxLrls = B  * (B'*B + size(obj.X,1) * lambda * eye(size(B,2)) ) * B'; 

            tmp = B  * (B'*B + size(obj.X,1) * lambda * eye(size(B,2)) );
            obj.approxLrls = sum(tmp.*B,2);
            
            
            % Compute probabilities associated with data points, according
            % to the computed approx scores
%             obj.P = zeros(1, size(obj.X,1));
            
            obj.P = obj.approxLrls./sum(obj.approxLrls);
            
            %%% Sample data points (rows of X) non-uniformly according to the LRLSs
%             sCount = 0;
%             i = 1;
%             obj.sampledPoints = zeros(1, chosenPar(1));
%             warning('Non-uniform sampling has no replacement');
%             while sCount < chosenPar(1)
%                 prob = rand;
%                 if prob <= obj.P(i)
%                     sCount = sCount+1;
%                     obj.sampledPoints(sCount) = i;
%                 end
%                 
%                 if (i<size(obj.X,1))
%                     i = i+1;
%                 else
%                     i = 1;
%                 end
%             end
            a = 1:size(obj.X,1);           %# possible numbers
            full = 0;
            obj.sampledPoints = [];
                        
            while full == 0
                obj.sampledPoints = [obj.sampledPoints a( sum( bsxfun(@ge, rand(chosenPar(1),1), cumsum(obj.P)'), 2) + 1 )];
                obj.sampledPoints = unique(obj.sampledPoints);
                if size(obj.sampledPoints,2) >= chosenPar(1)
                    full = 1;
                    obj.sampledPoints = obj.sampledPoints(1,1:chosenPar(1));
                end
            end
            
            obj.Xs = obj.X(obj.sampledPoints,:);
            
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
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.rng{obj.currentParIdx};
                available = true;
            end
        end        
    end
end
