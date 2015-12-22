classdef pegasos_dual_hinge_loss < filter
    %pegasos_kernel_hinge_loss Kernelized Pegasos subgradient descent iteration

    properties
        
        % Parameters used in the classical formulation
        X
        Y
        map                     % Kernel function in use
        mapPar                  % Kernel parameter(s)
        numSamples
        
        weights                 % Learned weights vector
        
        n                       % Number of samples
        sz                      % Size of the K or C matrix
        t                       % Number of outputs
        
        filterParGuesses        % Filter parameter guesses (range)
        numFilterParGuesses     % number of filter hyperparameters guesses
        
        lambda                  
        
        currentParIdx           % Current parameter combination indexes map container
        currentPar              % Current parameter combination map container
    end
    
    methods
        function obj = pegasos_dual_hinge_loss( map , mapPar , lambda, X , Y , numSamples , varargin)

            obj.init( map , mapPar , lambda, X , Y , numSamples , varargin);            
        end
        
        function init(obj , map , mapPar , lambda , X , Y , numSamples , varargin)
                 
            p = inputParser;
            
            %%%% Required parameters
            
            checkMap = @(x) true;
            checkMapPar = @(x) (x > 0);
            checkLambda = @(x) (x > 0);
            checkNumSamples = @(x) (x > 0);
            
            addRequired(p,'map',checkMap)
            addRequired(p,'mapPar',checkMapPar)
            addRequired(p,'lambda',checkLambda)
            addRequired(p,'X');
            addRequired(p,'Y');
            addRequired(p,'numSamples',checkNumSamples);
            
            %%%% Optional parameters
            % Optional parameter names:

            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x) x >= 0;

            defaultFilterParGuesses = [];
            
            defaultVerbose = 0;
            checkVerbose = @(x) (x==0) || (x==1);

            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses,checkNumFilterParGuesses)
            addParameter(p,'filterParGuesses',defaultFilterParGuesses)
            addParameter(p,'verbose',defaultVerbose,checkVerbose)
            
            % Parse function inputs
            parse(p, map , mapPar , lambda , X , Y , numSamples , varargin{:}{:})
                        
            % Get size of kernel/covariance matrix
            obj.sz = size(p.Results.X,1);
            
            % Get number of samples
            obj.n = p.Results.numSamples;

            % Store full kernel/covariance matrix
            obj.X = p.Results.X;
            obj.Y = p.Results.Y;

            % Store number of outputs
            obj.t = size(obj.Y,2);
            
            % Compute hyperparameter(s) range
            if ~isempty(p.Results.numFilterParGuesses) && isempty(p.Results.filterParGuesses)
                obj.numFilterParGuesses = p.Results.numFilterParGuesses;
                
            elseif isempty(p.Results.numFilterParGuesses) && ~isempty(p.Results.filterParGuesses)
                obj.filterParGuesses = p.Results.filterParGuesses;
                obj.numFilterParGuesses = size(p.Results.filterParGuesses,2);
                
            elseif ~isempty(p.Results.numFilterParGuesses) && ~isempty(p.Results.filterParGuesses)
                if p.Results.numFilterParGuesses == size( p.Results.filterParGuesses,2)
                    obj.filterParGuesses = p.Results.filterParGuesses;
                    obj.numFilterParGuesses = p.Results.numFilterParGuesses;
                else
                    error('numGuesses and fixedFilterParGuesses optional parameters are not consistent.');
                end
            end
            
            obj.weights = zeros(obj.n,obj.t);            
            obj.lambda = p.Results.lambda;
            obj.map = p.Results.map;
            obj.mapPar = p.Results.mapPar;
            
            obj.range();    % Compute range
            obj.currentParIdx = 0;
            obj.currentPar = [];   
            
            % Set verbosity          
            obj.verbose = p.Results.verbose;
        end
        
        % Compute range
        function obj = range(obj)            
            if isempty(obj.filterParGuesses)
                obj.filterParGuesses = 1:obj.numFilterParGuesses;
            end
        end
        
        % Perform the i-th step of subgradient descent
        function compute(obj )

            if (isempty(obj.currentPar))
                % If any current value for any of the parameters is not available, abort.
                error('Current filter parameter not available available. Exiting...');
            else
                if obj.verbose == 1
                    disp(['Iteration # ' , num2str(obj.currentPar)]);
                end
            end
            
            % Draw a training point randomly
            randIdx = randi(size(obj.X,1));
            xrand = obj.X(randIdx,:);
            yrand = obj.Y(randIdx,:);
            
            % Construct Kernel column
            argin = {};
            argin = [argin , 'mapParGuesses' , obj.mapPar];
            if ~isempty(obj.verbose)
                argin = [argin , 'verbose' , obj.verbose];
            end
            kernelCol = obj.map( xrand , obj.X , argin{:});
            kernelCol.next();
            % Compute kernel according to current hyperparameters
            kernelCol.compute();
            
            % Update weights (alpha) via subgradient descent
%             if (yrand/(obj.currentPar * obj.lambda)) * kernelCol.K * (obj.weights.*yrand)  < 1
            if (yrand/(obj.currentPar * obj.lambda)) * kernelCol.K * (obj.weights.*obj.Y)  < 1
               obj.weights(randIdx) = obj.weights(randIdx) + 1;   
%                obj.weights = obj.weights/norm(obj.weights);
            end
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            if isempty(obj.filterParGuesses)
                obj.range();
            end
            
            available = false;
            if length(obj.filterParGuesses) > obj.currentParIdx
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.filterParGuesses(:,obj.currentParIdx);
                available = true;
            end
        end
    end
end