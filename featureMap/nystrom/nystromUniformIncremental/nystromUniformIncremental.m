classdef nystromUniformIncremental < nystrom
    %NYSTROMUNIFORM Implementation of an integrated incremental Nystrom low-rank
    %approximator/regularizer and Tikhonov regularizer.
    %
    % Input parameters:
    % TO DO.

    
    properties
        numKerParRangeSamples   % Number of samples of X considered for estimating the maximum and minimum sigmas
        maxRank                 % Maximum rank of the kernel approximation
        
        filterParGuesses    % Vector of Tikhonov regularization parameter guesses
        
        fixedMapPar     % Fixed mapping parameter
        
        kernelType      % Type of approximated kernel
        sampledPoints   % Current sampled columns
        SqDistMat       % Squared distances matrix
        Xs              % Sampled points
        Y               % Training outputs
        C               % Current n-by-l matrix, composed of the evaluations of the kernel function for the sampled columns
        W               % Current l-by-l matrix, s. t. K ~ C * W^{-1} * C^T
        
        s               % Number of new columns
        ntr             % Total number of training samples
        
        prevPar
        
        A
        B
        D
        M
        alpha
    end
    
    methods
        % Constructor
        function obj = nystromUniformIncremental( X , Y , ntr , numNysParGuesses , numMapParGuesses , filterParGuesses , numKerParRangeSamples , maxRank , fixedMapPar , verbose)
            
            obj.init( X , Y , ntr , numNysParGuesses , numMapParGuesses , filterParGuesses , numKerParRangeSamples , maxRank , fixedMapPar , verbose);
            
            warning('Kernel type set by default to "gaussian"');
            obj.kernelType = @gaussianKernel;
        end
        
        % Initialization function
        function obj = init(obj , X , Y , ntr , numNysParGuesses , numMapParGuesses , filterParGuesses , numKerParRangeSamples , maxRank , fixedMapPar , verbose)
            
            obj.X = X;
            obj.Y = Y;
            obj.ntr =  ntr;
            obj.numKerParRangeSamples = numKerParRangeSamples;
            obj.d = size(X , 2);     
            obj.maxRank = maxRank;
            obj.fixedMapPar = fixedMapPar;
            obj.filterParGuesses = filterParGuesses;
            
            if ~isempty(fixedMapPar)
                obj.numMapParGuesses = 1;
            else
                obj.numMapParGuesses = numMapParGuesses;
            end
            
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
            
            % Compute range of number of sampled columns (m)
            tmpm = round(linspace(1, obj.maxRank , obj.numNysParGuesses));   
            
            % Approximated kernel parameter range
            
            if isempty(obj.fixedMapPar)
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
            else
                tmpKerPar = obj.fixedMapPar;
            end

            % Generate all possible parameters combinations
            [p,q] = meshgrid(tmpm, tmpKerPar);
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
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Incremental Update Rule %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%

            if isempty(obj.prevPar)
                
                %%% Initialization (i = 1)

                obj.D = cell(size(obj.filterParGuesses));
                obj.M = cell(size(obj.filterParGuesses));
                obj.alpha = cell(size(obj.filterParGuesses));
                            
                sampledPoints = 1:chosenPar(1);
                obj.s = chosenPar(1);  % Number of new columns
                obj.Xs = obj.X(sampledPoints,:);
                obj.computeSqDistMat(obj.X , obj.Xs);
                A = exp(-obj.SqDistMat / (2 * chosenPar(2)^2));
                obj.C = A;
                
                for i = 1:size(obj.filterParGuesses,2)
                    obj.M{i} = inv(A'*A + obj.ntr * obj.filterParGuesses(i) * eye(obj.s));
                    obj.D{i} = obj.M{i};
                    obj.alpha{i} = obj.D{i} * A' * obj.Y;
                end
                
            elseif obj.prevPar(1) ~= chosenPar(1)
                
                %%% Generic i-th incremental update step
                
                %Sample new columns of K
                sampledPoints = obj.prevPar(1)+1:chosenPar(1);
                obj.s = chosenPar(1) - obj.prevPar(1);  % Number of new columns
                XsNew = obj.X(sampledPoints,:);
                obj.Xs = [ obj.Xs ; XsNew ];
                obj.computeSqDistMat(obj.X , XsNew);
                A = exp(-obj.SqDistMat / (2 * chosenPar(2)^2));
                
                % Update B
                obj.B = obj.C' * A;
                % Update C
                obj.C = [obj.C A];
                
                for i = 1:size(obj.filterParGuesses,2)

                    % Update D_(t)
                    obj.D{i} = inv(A' * A - obj.B' * obj.M{i} * obj.B + obj.ntr * obj.filterParGuesses(i) * eye(obj.s));
                    
                    % Precompute BD and MBD, needed later
                    BD{i} = obj.B * obj.D{i};
                    MBD{i} = obj.M{i} * BD{i};

                    % Update alpha_(t+1)
                    obj.alpha{i} = [ obj.alpha{i} + MBD{i} * obj.B' * obj.alpha{i} - MBD{i} * A' * obj.Y ; ...
                                  -obj.D{i} * obj.B' * obj.alpha{i} + obj.D{i} * A' * obj.Y        ];

                    % Update M_(t+1)
%                     norm(obj.M{i} + MBD{i} * obj.B')
                    obj.M{i} = [obj.M{i} + MBD{i} * obj.B' * obj.M{i} , -MBD{i}; ...
                                -MBD{i}'                                obj.D{i}];                
                end
            end
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
