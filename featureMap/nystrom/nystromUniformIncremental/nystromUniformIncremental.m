classdef nystromUniformIncremental < nystrom
    %NYSTROMUNIFORM Implementation of an integrated incremental Nystrom low-rank
    %approximator/regularizer and Tikhonov regularizer.
    %
    % Input parameters:
    % TO DO.

    
    properties
        numMapParRangeSamples   % Number of samples of X considered for estimating the maximum and minimum sigmas
        maxRank                 % Maximum rank of the kernel approximation
        
        filterParGuesses        % Filter parameter guesses
        numFilterParGuesses     % Number of filter parameter guesses
        
        mapParGuesses           % mapping parameter guesses
        numMapParGuesses        % Number of mapping parameter guesses
        
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
        
        % Variables for computeSqDistMat efficiency
        X1
        X2
        Sx1
        Sx2
        
        M
        alpha
    end
    
    methods
        % Constructor
%         function obj = nystromUniformIncremental( X , Y , ntr , numNysParGuesses , numMapParGuesses , filterParGuesses , numKerParRangeSamples , maxRank , fixedMapPar , verbose)
%             
%             obj.init( X , Y , ntr , numNysParGuesses , numMapParGuesses , filterParGuesses , numKerParRangeSamples , maxRank , fixedMapPar , verbose);
%             
%             warning('Kernel type set by default to "gaussian"');
%             obj.kernelType = @gaussianKernel;
%         end
        
        function obj = nystromUniformIncremental( X1 , X2 , ntr , varargin)
            obj.init( X1 , X2 , ntr , varargin);
        end
        
        % Initialization function
%         function obj = init(obj , X , Y , ntr , numNysParGuesses , numMapParGuesses , filterParGuesses , numKerParRangeSamples , maxRank , fixedMapPar , verbose)
%             
%             obj.X = X;
%             obj.Y = Y;
%             obj.ntr =  ntr;
%             obj.numKerParRangeSamples = numKerParRangeSamples;
%             obj.d = size(X , 2);     
%             obj.maxRank = maxRank;
%             obj.fixedMapPar = fixedMapPar;
%             obj.filterParGuesses = filterParGuesses;
%             
%             if ~isempty(fixedMapPar)
%                 obj.numMapParGuesses = 1;
%             else
%                 obj.numMapParGuesses = numMapParGuesses;
%             end
%             
%             obj.numNysParGuesses = numNysParGuesses;
%             
%             obj.verbose = 0;
%             if verbose == 1
%                 obj.verbose = 1;
%             end
%             
%             % Compute range
%             obj.range();
%             obj.currentParIdx = 0;
%             obj.currentPar = [];
%             obj.prevPar = [];
%         end

        function obj = init(obj , X , Y , ntr , varargin)
            

            p = inputParser;
            
            %%%% Required parameters
            % X , Y , ntr

            checkX = @(x) size(x,1) > 0 && size(x,2) > 0;
            checkY = @(x) size(x,1) > 0 && size(x,2) > 0;
            checkNtr = @(x) x > 0;
            
            addRequired(p,'X',checkX);
            addRequired(p,'Y',checkY);
            addRequired(p,'ntr',checkNtr);
            
 
            %%%% Optional parameters
            % Optional parameter names:
            % numNysParGuesses , maxRank , numMapParGuesses , mapParGuesses , filterParGuesses , numMapParRangeSamples  , verbose
            
            % numNysParGuesses       % Cardinality of number of samples for Nystrom approximation parameter guesses
            defaultNumNysParGuesses = [];
            checkNumNysParGuesses = @(x) x > 0 ;            
            addParameter(p,'numNysParGuesses',defaultNumNysParGuesses,checkNumNysParGuesses);                    
            
            % maxRank        % Maximum rank of the Nystrom approximation
            defaultMaxRank = [];
            checkMaxRank = @(x) x > 0 ;            
            addParameter(p,'maxRank',defaultMaxRank,checkMaxRank);        
            
            % numMapParGuesses        % Number of map parameter guesses
            defaultNumMapParGuesses = [];
            checkNumMapParGuesses = @(x) x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses);        
            
            % mapParGuesses        % Map parameter guesses
            defaultMapParGuesses = [];
            checkMapParGuesses = @(x) size(x,1) > 0 && size(x,2) > 0 ;      
            addParameter(p,'mapParGuesses',defaultMapParGuesses,checkMapParGuesses);        
            
            % filterParGuesses       % Filter parameter guesses
            defaultFilterParGuesses = [];
            checkFilterParGuesses = @(x) size(x,1) > 0 && size(x,2) > 0 ;            
            addParameter(p,'filterParGuesses',defaultFilterParGuesses,checkFilterParGuesses);                    
            
            % numMapParRangeSamples        % Number of map parameter guesses
            defaultNumMapParRangeSamples = [];
            checkNumMapParRangeSamples = @(x) x > 0 ;            
            addParameter(p,'numMapParRangeSamples',defaultNumMapParRangeSamples,checkNumMapParRangeSamples);            

            % verbose             % 1: verbose; 0: silent      
            defaultVerbose = 0;
            checkVerbose = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'verbose',defaultVerbose,checkVerbose);
            
            % Parse function inputs
            if isempty(varargin{:})
                parse(p, X , Y , ntr)
            else
                parse(p, X , Y , ntr , varargin{:}{:})
            end
            
            % Assign parsed parameters to object properties
            fields = fieldnames(p.Results);
%             fieldsToIgnore = {'X1','X2'};
%             fields = setdiff(fields, fieldsToIgnore);
            for idx = 1:numel(fields)
                obj.(fields{idx}) = p.Results.(fields{idx});
            end
            
            % Joint parameters parsing
            if isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                error('either mapParGuesses or numMapParGuesses must be specified');
            end    
            
            if (~isempty(obj.mapParGuesses)) && (~isempty(obj.numMapParGuesses)) && (size(obj.mapParGuesses,2) ~= obj.numMapParGuesses)
                error('The size of mapParGuesses and numMapParGuesses are different');
            end
            
            if ~isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                obj.numMapParGuesses = size(obj.mapParGuesses,2);
            end
            
            if size(X,1) ~= size(Y,1)
                error('X and Y have incompatible sizes');
            end
            
            obj.d = size(X , 2);
            
            warning('Kernel used by nystromUniformIncremental is set to @gaussianKernel');
            obj.kernelType = @gaussianKernel;

            % Conditional range computation
%             if isempty(obj.mapParGuesses)
                display('Computing range');
                obj.range();    % Compute range
%             end
            obj.currentParIdx = 0;
            obj.currentPar = [];
        end

        function obj = range(obj)
            
            % Compute range of number of sampled columns (m)
            tmpm = round(linspace(1, obj.maxRank , obj.numNysParGuesses));   

            % Approximated kernel parameter range
            
            if isempty(obj.mapParGuesses)
                % Compute max and min sigma guesses

                % Extract an even number of samples without replacement                

                % WARNING: not compatible with versions older than 2014
                %samp = datasample( obj.X(:,:) , obj.numKerParRangeSamples - mod(obj.numKerParRangeSamples,2) , 'Replace', false);

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

                firstPercentile = round(0.01 * numel(D) + 0.5);
                minGuess = sqrt( D(firstPercentile));
                maxGuess = sqrt( max(D) );

                if minGuess <= 0
                    minGuess = eps;
                end
                if maxGuess <= 0
                    maxGuess = eps;
                end	

                tmpMapPar = linspace(minGuess, maxGuess , obj.numMapParGuesses);
            else
                tmpMapPar = obj.mapParGuesses;
            end

            % Generate all possible parameters combinations
            [p,q] = meshgrid(tmpm, tmpMapPar);
            tmp = [p(:) q(:)]';
%             obj.rng = num2cell(tmp , 1);
            obj.rng = tmp;
        end
        
        % Computes the squared distance matrix SqDistMat based on X1, X2
        function computeSqDistMat(obj , X1 , X2)
            
            if ~isempty(X1)
                obj.X1 = X1;
                obj.Sx1 = sum( X1.*X1 , 2);
            end
            if ~isempty(X2)
                obj.X2 = X2;
                obj.Sx2 = sum( X2.*X2 , 2)';
            end
            if ~isempty(X1) && ~isempty(X2)
                Sx1x2 = X1 * X2';
                obj.SqDistMat = repmat(obj.Sx1 , 1 , size(X2,1)) -2*Sx1x2 + repmat(obj.Sx2 , size(X1,1) , 1);
            elseif isempty(X1) && ~isempty(X2)
                Sx1x2 = obj.X1 * X2';
                obj.SqDistMat = repmat(obj.Sx1 , 1 , size(X2,1)) -2*Sx1x2 + repmat(obj.Sx2 , size(obj.X1,1) , 1);
            elseif ~isempty(X1) && isempty(X2)
                Sx1x2 = X1 * obj.X2';
                obj.SqDistMat = repmat(obj.Sx1 , 1 , size(obj.X2,1)) -2*Sx1x2 + repmat(obj.Sx2 , size(X1,1) , 1);
            else
                Sx1x2 = obj.X1 * obj.X2';
                obj.SqDistMat = repmat(obj.Sx1 , 1 , size(obj.X2,1)) -2*Sx1x2 + repmat(obj.Sx2 , size(obj.X1,1) , 1);
            end
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

            if obj.currentParIdx == 1
                
                %%% Initialization (i = 1)
                
                % Preallocate calls of matrices 
                
                obj.M = cell(size(obj.filterParGuesses));
                [obj.M{:,:}] = deal(zeros(obj.maxRank));
                
                obj.alpha = cell(size(obj.filterParGuesses));
                [obj.alpha{:,:}] = deal(zeros(obj.maxRank,size(obj.Y,2)));
                
                sampledPoints = 1:chosenPar(1);
                obj.s = chosenPar(1);  % Number of new columns
                obj.Xs = obj.X(sampledPoints,:);
                obj.computeSqDistMat(obj.X , obj.Xs);
                A = exp(-obj.SqDistMat / (2 * chosenPar(2)^2));
                
                A = A/sqrt(obj.ntr);

                % Set C_2
                obj.C = A;

                for i = 1:size(obj.filterParGuesses,2)
                    % D_1
                    D = inv(A'*A + obj.filterParGuesses(i) * eye(obj.s));
                    % alpha_2
                    obj.alpha{i} = 	D * (A' * obj.Y / sqrt(obj.ntr));
                    % M_2
                    obj.M{i}(1:chosenPar(1), 1:chosenPar(1)) = D;
                end
                
            elseif obj.prevPar(1) ~= chosenPar(1)
               
                %%% Generic i-th incremental update step
                
                %Sample new columns of K
                sampledPoints = (obj.prevPar(1)+1):chosenPar(1);
                obj.s = chosenPar(1) - obj.prevPar(1);  % Number of new columns
                XsNew = obj.X(sampledPoints,:);
                obj.Xs = [ obj.Xs ; XsNew ];
                
                if isempty(obj.Sx1)
                    obj.computeSqDistMat(obj.X , XsNew);
                else
                    obj.computeSqDistMat([] , XsNew);
                end
                
                A = exp(-obj.SqDistMat / (2 * chosenPar(2)^2));
               
                A = A/sqrt(obj.ntr);
                
                % Update B_(t)
                B = obj.C' * A;
                
                % Update C_(t+1)
                obj.C = [obj.C A];
                
                Aty = A' * obj.Y/sqrt(obj.ntr);
                
                % for cycle implementation
                
                for i = 1:size(obj.filterParGuesses,2)

%                     max(max(obj.M{i}(1:obj.prevPar(1), 1:obj.prevPar(1))))
                    MB = obj.M{i}(1:obj.prevPar(1), 1:obj.prevPar(1)) * B;

                    %D = inv(A' * A - B' * MB +  obj.filterParGuesses(i) * eye(obj.s));

%                     sum(sum(isinf(A' * A - B' * MB)))
%                     sum(sum(isnan(A' * A - B' * MB)))
                    
%                     isreal(A)
%                     isreal(B)
%                     isreal(MB)
%                     sum(sum(isnan(A)))
%                     sum(sum(isnan(B)))
%                     sum(sum(isnan(MB)))  
%                     sum(sum(isinf(A)))
%                     sum(sum(isinf(B)))
%                     sum(sum(isinf(MB)))     
%                     if  ~isreal(A) || ~isreal(B) ||~isreal(MB) || sum(sum(isnan(A))) > 0 || sum(sum(isnan(B)))> 0 || sum(sum(isnan(MB)))> 0 || sum(sum(isinf(A)))> 0 ||sum(sum(isinf(B)))> 0 || sum(sum(isinf(MB)))    
%                         
%                     end
                    [U, S] = eig(full(A' * A - B' * MB));
                    ds = diag(S);
                    ds = (ds>0).*ds;    % Set eigenvalues < 0 for numerical reasons to 0
                    ds = real((ds>0).*ds);    % Set eigenvalues < 0 for numerical reasons to 0
                    U = real(U);
                    D = U * diag(1./(ds + obj.filterParGuesses(i))) * U';
%                     isreal(D)
%                     isreal(U)
%                     isreal(diag(1./(ds + obj.filterParGuesses(i))))
                    
                    MBD = MB * D;
%                     isreal(MBD)

                    df = B' * obj.alpha{i} - Aty;
                    obj.alpha{i}(1:obj.prevPar(1),:) = obj.alpha{i} + MBD * df; 
                    obj.alpha{i}((obj.prevPar(1)+1):chosenPar(1),:) =  -D * df;
                  
                    obj.M{i}(1:obj.prevPar(1), 1:obj.prevPar(1)) = obj.M{i}(1:obj.prevPar(1), 1:obj.prevPar(1)) + MBD*MB';
                    obj.M{i}(1:obj.prevPar(1), (obj.prevPar(1)+1):chosenPar(1)) = -MBD;
                    obj.M{i}((obj.prevPar(1)+1):chosenPar(1), 1:obj.prevPar(1)) = -MBD';
                    obj.M{i}((obj.prevPar(1)+1):chosenPar(1), (obj.prevPar(1)+1):chosenPar(1)) = D;
                end
            end
        end
        
        function resetPar(obj)
            
            obj.currentParIdx = 0;
            obj.currentPar = [];
        
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            % If any range for any of the parameters is not available, recompute all ranges.
%             if cellfun(@isempty,obj.mapParGuesses)
%                 obj.range();
%             end
            if isempty(obj.rng)
                obj.range();
            end
            
            available = false;
%             if length(obj.mapParGuesses) > obj.currentParIdx
%                 obj.currentParIdx = obj.currentParIdx + 1;
%                 obj.currentPar = obj.mapParGuesses{obj.currentParIdx};
%                 available = true;
%             end
            if length(obj.rng) > obj.currentParIdx
                obj.prevPar = obj.currentPar;
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.rng(:,obj.currentParIdx);
                available = true;
            end
        end
    end
end
