classdef subGD_dual_hinge_loss < filter
    % Subgradient descent iterative filter for the hinge loss
    %   Detailed explanation goes here

    properties
        
        % Parameters used in the classical formulation
        K
        Y
        
        weights                 % Learned weights vector
        Avec
        
        n                       % Number of samples
        sz                      % Size of the K or C matrix
        t                       % Number of outputs
        
        filterParGuesses        % Filter parameter guesses (range)
        numFilterParGuesses     % number of filter hyperparameters guesses
        
        eta                     % Step size
        theta                   % Step size sequence exponent
        
        currentParIdx           % Current parameter combination indexes map container
        currentPar              % Current parameter combination map container
    end
    
    methods
      
        function obj = subGD_dual_hinge_loss( K , Y , numSamples , varargin)

            obj.init(  K , Y , numSamples , varargin );            
        end
        
        function init(obj , K , Y , numSamples , varargin)
                 
            p = inputParser;
            
            %%%% Required parameters
            
            checkK = @(x) size(x,1) == size(x,2);
            checkNumSamples = @(x) (x > 0);
            
            addRequired(p,'K',checkK);
            addRequired(p,'Y');
            addRequired(p,'numSamples',checkNumSamples);
            
            %%%% Optional parameters
            % Optional parameter names:

%             defaultEta = 1/sqrt(2);
            defaultEta = 1/4;
            checkEta = @(x) x > 0;

            defaultTheta = 1/2;
            checkTheta = @(x) (x <= 1 && x >= 0);
            
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
            parse(p, K , Y , numSamples , varargin{:}{:})
                        
            % Get size of kernel/covariance matrix
            obj.sz = size(p.Results.K,1);
            
            % Get number of samples
            obj.n = p.Results.numSamples;

            % Store full kernel/covariance matrix
            obj.K = p.Results.K;
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

            obj.eta = p.Results.eta;
            obj.theta = p.Results.theta;
                        
            obj.range();    % Compute range
            obj.currentParIdx = 0;
            obj.currentPar = [];   
            
            % Set verbosity          
            obj.verbose = p.Results.verbose;
        end
        
        function obj = range(obj)
            
            if isempty(obj.filterParGuesses)
                obj.filterParGuesses = 1:obj.numFilterParGuesses;
            end
        end
        
        function compute(obj )

            if (isempty(obj.currentPar))
                % If any current value for any of the parameters is not available, abort.
                error('Current filter parameter not available available. Exiting...');
            else
                if obj.verbose == 1
                    disp(['Iteration # ' , num2str(obj.currentPar)]);
                end
            end
            
            % Compute current prediction
            Ypred = obj.K * obj.weights;
            
            % Get wrong predictions indexes
            mask = (Ypred .* obj.Y <= 1);
            
            % Compute GD iteration step
%             step = (1/obj.n) * obj.eta * obj.currentPar^(-obj.theta);
            step = obj.eta * obj.currentPar^(-obj.theta);
            
            % Update weights
%             obj.weights = obj.weights + step * obj.K * (mask .* obj.Y);

%             obj.weights = obj.weights + step * mask .* obj.Y;
            obj.weights = obj.weights + step * (1/obj.n) * mask .* obj.Y;
            
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