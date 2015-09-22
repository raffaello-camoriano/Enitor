classdef gaussianKernel < kernel
    %GAUSSIAN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        numMapParGuesses        % Number of guesses for the parameters
        mapParGuesses           % Parameter ranges container
        currentParIdx           % Current parameter combination indexes map container
        currentPar              % Current parameter combination map container
        
        n               % Number of X1 samples
        m               % Number of X2 samples
        SqDistMat       % n-by-m squared distances matrix 

    end
    
    methods
        % Construct a size(X1,1) * size(X2,1) Gaussian kernel object 
        function obj = gaussianKernel( X1 , X2 , varargin)
            obj.init( X1 , X2 , varargin);
        end
        
        function init( obj , X1 , X2 , varargin)
            
            p = inputParser;
            
            %%%% Required parameters
            
            checkX1 = @(x) size(x,1) > 0 && size(x,2) > 0;
            checkX2 = @(x) size(x,1) > 0 && size(x,2) > 0;
            
            addRequired(p,'X1',checkX1);
            addRequired(p,'X2',checkX2);
            
            %%%% Optional parameters
            % Optional parameter names:
            % numMapParGuesses , mapParGuesses, verbose
            
            % mapParGuesses       % Map parameter guesses cell array
            defaultMapParGuesses = [];
            checkMapParGuesses = @(x) ismatrix(x) && size(x,2) > 0 ;            
            addParameter(p,'mapParGuesses',defaultMapParGuesses,checkMapParGuesses);                    
            
            % numMapParGuesses        % Number of map parameter guesses
            defaultNumMapParGuesses = [];
            checkNumMapParGuesses = @(x) x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses);        
            
            % verbose             % 1: verbose; 0: silent      
            defaultVerbose = 0;
            checkVerbose = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'verbose',defaultVerbose,checkVerbose);
            
            % Parse function inputs
            if isempty(varargin)
                parse(p, X1 , X2)
            else
                parse(p, X1 , X2 , varargin{:}{:})
            end
            
            % Assign parsed parameters to object properties
            fields = fieldnames(p.Results);
            fieldsToIgnore = {'X1','X2'};
            fields = setdiff(fields, fieldsToIgnore);
            for idx = 1:numel(fields)
                obj.(fields{idx}) = p.Results.(fields{idx});
            end
            
            %%% Joint parameters validation
            if size(X1,2) ~= size(X2,2)
                error('size(X1,2) ~= size(X2,1)');
            end
%             
            if isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                error('either mapParGuesses or numMapParGuesses must be specified');
            end    
%             
%             if ~isempty(obj.mapParGuesses) && ~isempty(obj.numMapParGuesses)
%                 error('mapParGuesses and numMapParGuesses cannot be specified together');
%             end
            
            if (~isempty(obj.mapParGuesses)) && (~isempty(obj.numMapParGuesses)) && (size(obj.mapParGuesses,2) ~= obj.numMapParGuesses)
                error('The size of mapParGuesses and numMapParGuesses are different');
            end
            
            if size(X2,2) ~= size(X1,2)
                error('X1 and X2 have incompatible sizes');
            end
            
            % Set dimensions and compute square distances matrix
            obj.n = size(X1 , 1);
            obj.m = size(X2 , 1);
            obj.computeSqDistMat(X1,X2);
            
            % Conditional range computation
            if isempty(obj.mapParGuesses)
                if obj.verbose == 1
                    display('Computing Gaussian kernel range');
                end
                obj.range();    % Compute range
            end
            obj.currentParIdx = 0;
            obj.currentPar = [];
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

            %             firstPercentile = round(0.01*numel(D)+0.5);
%             minGuess = sqrt(D(firstPercentile));
%             maxGuess = sqrt(max(max(obj.SqDistMat)));
            
            
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
            
            tmp = linspace(minGuess, maxGuess , obj.numMapParGuesses);
%             obj.mapParGuesses = num2cell(tmp);
            obj.mapParGuesses = tmp;
        end
        Xtrain 
        % Computes the kernel matrix K based on SqDistMat and
        % kernel parameters
        function compute(obj , kerPar)
            if( nargin > 1 )
                obj.K = exp(-obj.SqDistMat / (2 * kerPar(1)^2));
            
            % If any current value for any of the parameters is not available, abort.
            elseif (nargin == 1) && (isempty(obj.currentPar))
                error('Kernel parameter(s) not explicitly specified, and no internal current parameter available. Exiting...');
            else
                if obj.verbose == 1
                    disp('Kernel will be computed according to the internal current hyperparameter(s)');
                    obj.currentPar
                end

                obj.K = exp(-obj.SqDistMat / (2 * obj.currentPar(1)^2));
            end
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            % If any range for any of the parameters is not available, recompute all ranges.
%             if cellfun(@isempty,obj.mapParGuesses)
%                 obj.range();
%             end
            if isempty(obj.mapParGuesses)
                obj.range();
            end
            
            available = false;
%             if length(obj.mapParGuesses) > obj.currentParIdx
%                 obj.currentParIdx = obj.currentParIdx + 1;
%                 obj.currentPar = obj.mapParGuesses{obj.currentParIdx};
%                 available = true;
%             end
            if length(obj.mapParGuesses) > obj.currentParIdx
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.mapParGuesses(:,obj.currentParIdx);
                available = true;
            end
        end
    end
end
