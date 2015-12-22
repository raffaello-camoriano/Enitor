classdef SsubGD_primal_hinge_loss < filter
    %subGD_primal_hinge_loss Stochastic linear subgradient descent with the
    %hinge loss (1 pass over data)
    properties
        
        % Parameters used in the classical formulation
        X
        Y
        
        weights                 % Learned weights vector
        Avec
        
        n                       % Number of samples
        d                       % samples dimensionality
        sz                      % Size of the K or C matrix
        t                       % Number of outputs
        
        filterParGuesses        % Filter parameter guesses (range)
        numFilterParGuesses     % number of filter hyperparameters guesses
        
        eta                     % Step size
        theta                   % Step size decreasing sequence factor exponent
        
        randOrd
        
        currentParIdx           % Current parameter combination indexes map container
        currentPar              % Current parameter combination map container
    end
    
    methods
      
        function obj = SsubGD_primal_hinge_loss( X , Y , numSamples , varargin)

            obj.init( X , Y , numSamples , varargin );            
        end
        
        function init(obj , X , Y , numSamples , varargin)
                 
            p = inputParser;
            
            %%%% Required parameters
            
            checkNumSamples = @(x) (x > 0);
            
            addRequired(p,'X');
            addRequired(p,'Y');
            addRequired(p,'numSamples',checkNumSamples);
            
            %%%% Optional parameters
            % Optional parameter names:

            defaultEta = 1/4;
            checkEta = @(x) x > 0;

            defaultTheta = -1/2;
            checkTheta = @(x) (x <= 0 && x >= -1);

            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x) x >= 0;

            defaultFilterParGuesses = [];
            
            defaultVerbose = 0;
            checkVerbose = @(x) (x==0) || (x==1);

            addParameter(p,'eta',defaultEta,checkEta)
            addParameter(p,'theta',defaultTheta,checkTheta)
            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses,checkNumFilterParGuesses)
            addParameter(p,'filterParGuesses',defaultFilterParGuesses)
            addParameter(p,'verbose',defaultVerbose,checkVerbose)
            
            % Parse function inputs
            parse(p, X , Y , numSamples , varargin{:}{:})
              
            % Get number of samples
            obj.n = p.Results.numSamples;
            
            % Get dimensionality of samples
            obj.d = size(X,2);

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
            
            obj.weights = zeros(obj.d,obj.t);
            
            obj.eta = p.Results.eta;
            obj.theta = p.Results.theta;
            
            obj.range();    % Compute range
            obj.currentParIdx = 0;
            obj.currentPar = [];   
            
            % Set verbosity          
            obj.verbose = p.Results.verbose;
            
            % Draw a new random ordering
			obj.randOrd = randperm(size(obj.X,1));
        end
        
        function obj = range(obj)
            
            if isempty(obj.filterParGuesses)
                obj.filterParGuesses = 1:obj.numFilterParGuesses;
            end
        end
        
        function compute(obj)

            if (isempty(obj.currentPar))
                % If any current value for any of the parameters is not available, abort.
                error('Current filter parameter not available available. Exiting...');
            else
                if obj.verbose == 1
                    disp(['Iteration # ' , num2str(obj.currentPar)]);
                end
            end
            
            randIdx = obj.randOrd(mod(obj.currentPar-1,obj.n) + 1);
            
            Ypred = obj.X(randIdx,:) * obj.weights;
            if (Ypred * obj.Y(randIdx,:) < 1)
				obj.weights = obj.weights + obj.eta * obj.currentPar^obj.theta * obj.X(randIdx,:)' * obj.Y(randIdx,:);
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
