classdef gaussianKernel < kernel & matlab.mixin.Copyable
    %GAUSSIAN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        rng             % Parameter ranges cell array 
        numGuesses      % Number of guesses for the parameters
        currentParIdx   % Current parameter combination indexes
        
        SqDistMat
        n
        m
    end
    
    methods
        
        function obj = gaussianKernel( X1 , X2 , numGuesses , sigma )

            if  nargin > 0
                if  nargin > 3
                    obj.init( X1 , X2 , numGuesses, sigma );
                elseif nargin > 2
                    obj.init( X1 , X2 , numGuesses);
                else
                    obj.init( X1 , X2);
                end
            end
        end
        
        function init( obj , X1 , X2 , numGuesses , sigma )
            
            if( nargin > 3 )
                if numGuesses > 0
                    obj.numGuesses = numGuesses;
                else
                    obj.numGuesses = 1;
                end            

                % Initialize range map
                rangeKeySet = {'sigma'};
                rangeValueSet = cell(size(rangeKeySet,1));
                rangeValueSet{:,:} = zeros(obj.numGuesses,1);
                obj.rng = containers.Map(rangeKeySet,rangeValueSet);

                % Initialize current parameter combination indexes
                currentParIdxKeySet = rangeKeySet;
                currentParIdxValueSet = cell(size(currentParIdxKeySet,1));
                currentParIdxValueSet{:,:} = 0;
                obj.currentParIdx = containers.Map(currentParIdxKeySet,currentParIdxValueSet);
            end
            
            obj.n = size(X1 , 1);
            obj.m = size(X2 , 1);
            obj.computeSqDistMat(X1,X2);
            
            if  nargin > 4
                obj.compute(sigma);
            end
        end
        
        % Computes the squared distance matrix SqDistMat based on X1, X2
        function computeSqDistMat(obj , X1 , X2)
            
            Sx1 = sum( X1.*X1 , 2);
            Sx2 = sum( X2.*X2 , 2)';
            Sx1x2 = X1 * X2';
            
            obj.SqDistMat = repmat(Sx1 , 1 , obj.m) -2*Sx1x2 + repmat(Sx2 , obj.n , 1);
        
        end
        
        % Computes the kernel matrix SqDistMat based on SqDistMat and sigma
        function compute(obj , sigma)
            if( nargin > 1 )
                obj.K = exp(-obj.SqDistMat/(2*sigma^2));
            else
                disp('sigma parameter not specified! Exiting...');
            end
        end
        
        % Computes the range for the sigma parameter guesses
        function obj = range(obj)
            
            if (obj.m ~= obj.n)
                error('Error, the distance matrix is not squared! Aborting...');
            end
            
            % Compute max and min sigma guesses (same strategy used by
            % GURLS)
            
            D = sort(obj.SqDistMat(tril(true(obj.n),-1)));
            firstPercentile = round(0.01*numel(D)+0.5);
            minGuess = sqrt(D(firstPercentile));
            maxGuess = sqrt(max(max(obj.SqDistMat)));

            if minGuess <= 0
                minGuess = eps;
            end
            if maxGuess <= 0
                maxGuess = eps;
            end	
            
            obj.rng('sigma') = linspace(minGuess, maxGuess , obj.numGuesses);
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)
            
            % If any range for any of the parameters is not available, recompute all ranges.
            if sum(cellfun(@isempty,values(obj.rng))) == 0
                obj.range
            end
            
            %idxToBeIncremented = cellfun(@(x,y) length(x) >= y , values(obj.rng) , obj.currentParIdx + 1);
            
            available = false;
            for key = keys(obj.rng)
                keyStr = key{1};
                if length(obj.rng(keyStr)) >= obj.currentParIdx(keyStr) + 1
                    obj.currentParIdx(keyStr) = obj.currentParIdx(keyStr) + 1;
                    available = true;
                end
            end
            
%             if length(obj.rng('sigma')) >= (obj.currentParIdx('sigma') + 1)
%                 obj.currentParIdx('sigma') = obj.currentParIdx('sigma') + 1;
%                 available = true;
%             else
%                 available = false;
%             end

        end
    end
end
