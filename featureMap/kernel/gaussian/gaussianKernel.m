classdef gaussianKernel < kernel
    %GAUSSIAN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        numGuesses      % Number of guesses for the parameters
        rng             % Parameter ranges map container
        currentParIdx   % Current parameter combination indexes map container
        currentPar      % Current parameter combination map container
        
        SqDistMat
        n
        m
    end
    
    methods
        
        function obj = gaussianKernel( X1 , X2 , numGuesses )

            if  nargin > 0
                if  nargin > 2
                    obj.init( X1 , X2 , numGuesses);
                else
                    obj.init( X1 , X2);
                end
            end
        end
        
        function init( obj , X1 , X2 , numGuesses)
            
            % Set dimensions and compute square distances matrix
            obj.n = size(X1 , 1);
            obj.m = size(X2 , 1);
            obj.computeSqDistMat(X1,X2);
            
            if( nargin == 4 )
                if numGuesses > 0
                    obj.numGuesses = numGuesses;
                else
                    obj.numGuesses = 1;
                end            

%                 % Initialize range map
%                 rangeKeySet = {'sigma'};
%                 rangeValueSet = cell(size(rangeKeySet,1));
%                 rangeValueSet{:,:} = zeros(obj.numGuesses,1);
%                 obj.rng = containers.Map(rangeKeySet,rangeValueSet);
% 
%                 % Initialize current parameter combination indexes map
%                 currentParIdxKeySet = rangeKeySet;
%                 currentParIdxValueSet = cell(size(currentParIdxKeySet,1));
%                 currentParIdxValueSet{:,:} = 0;
%                 obj.currentParIdx = containers.Map(currentParIdxKeySet,currentParIdxValueSet);
%                 
%                 % Initialize current parameter combination map
%                 currentParKeySet = rangeKeySet;
%                 currentParValueSet = cell(size(currentParIdxKeySet,1));
%                 obj.currentPar = containers.Map(currentParKeySet,currentParValueSet);
                
                obj.range();    % Compute range
                obj.currentParIdx = 0;
                obj.currentPar = [];
                
            end
        end
        
        % Computes the squared distance matrix SqDistMat based on X1, X2
        function computeSqDistMat(obj , X1 , X2)
            
            Sx1 = sum( X1.*X1 , 2);
            Sx2 = sum( X2.*X2 , 2)';
            Sx1x2 = X1 * X2';
            
            obj.SqDistMat = repmat(Sx1 , 1 , obj.m) -2*Sx1x2 + repmat(Sx2 , obj.n , 1);
        
        end
                
        % Computes the range for the hyperparameter guesses
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
            
            tmp = linspace(minGuess, maxGuess , obj.numGuesses);
            obj.rng = num2cell(tmp);
        end
%         % Computes the range for the hyperparameter guesses
%         function obj = range(obj)
%             
%             if (obj.m ~= obj.n)
%                 error('Error, the distance matrix is not squared! Aborting...');
%             end
%             
%             % Compute max and min sigma guesses (same strategy used by
%             % GURLS)
%             
%             D = sort(obj.SqDistMat(tril(true(obj.n),-1)));
%             firstPercentile = round(0.01*numel(D)+0.5);
%             minGuess = sqrt(D(firstPercentile));
%             maxGuess = sqrt(max(max(obj.SqDistMat)));
% 
%             if minGuess <= 0
%                 minGuess = eps;
%             end
%             if maxGuess <= 0
%                 maxGuess = eps;
%             end	
%             
%             obj.rng('sigma') = linspace(minGuess, maxGuess , obj.numGuesses);
%         end
        
        % Computes the kernel matrix SqDistMat based on SqDistMat and
        % kernel parameters
        function compute(obj , kerPar)
            if( nargin > 1 )
                obj.K = exp(-obj.SqDistMat / (2 * kerPar(1)^2));
            
            % If any current value for any of the parameters is not available, abort.
            elseif (nargin == 1) && (isempty(obj.currentPar))
                error('Kernel parameter(s) not explicitly specified, and no internal current parameter available. Exiting...');
            else
                disp('Kernel will be computed according to the internal current hyperparameter(s)');
                obj.currentPar
                obj.K = exp(-obj.SqDistMat / (2 * obj.currentPar(1)^2));
            end
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            % If any range for any of the parameters is not available, recompute all ranges.
            if cellfun(@isempty,obj.rng)
                obj.range();
            end

            available = false;
            if length(obj.rng) > obj.currentParIdx
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.rng{obj.currentParIdx};
                available = true;
            end
        end

%         % returns true if the next parameter combination is available and
%         % updates the current parameter combination 'currentPar'
%         function available = next(obj)
%             
%             % If any range for any of the parameters is not available, recompute all ranges.
%             if sum(cellfun(@isempty,values(obj.rng))) > 0
%                 obj.range
%             end
%                         
%             available = false;
%             for key = keys(obj.rng)
%                 keyStr = key{1};
%                 if length(obj.rng(keyStr)) >= obj.currentParIdx(keyStr) + 1
%                     obj.currentParIdx(keyStr) = obj.currentParIdx(keyStr) + 1;
%                     
%                     tmp = obj.rng(keyStr);
%                     obj.currentPar(keyStr) = tmp(obj.currentParIdx(keyStr));
%                     
%                     available = true;
%                 end
%             end
%         end
    end
end
