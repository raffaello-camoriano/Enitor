classdef gdesc_kernel_hinge_loss < filter
    %TIKHONOV Summary of this class goes here
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
        
        gamma                   % Step size
        
        currentParIdx           % Current parameter combination indexes map container
        currentPar              % Current parameter combination map container
    end
    
    methods
      
        function obj = gdesc_kernel_hinge_loss( K , Y , numSamples , varargin)

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

            defaultGamma = 1/(4*numSamples);
%             defaultGamma = 1/(8*numSamples);
            checkGamma = @(x) x > 0;

            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x) x >= 0;

            defaultFilterParGuesses = [];
            
            defaultVerbose = 0;
            checkVerbose = @(x) (x==0) || (x==1);

            addParameter(p,'gamma',defaultGamma,checkGamma)
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
            obj.Avec = zeros(obj.n,obj.t);
            
            obj.gamma = p.Results.gamma;
            
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
            
            Ypred = obj.K * obj.weights;
            mask = (sign(Ypred) ~= obj.Y);
            SG = -mask.*obj.Y;                    
%             SG = -mask.*(obj.K*obj.Y);                    
            obj.weights = obj.weights - obj.gamma * obj.currentPar^(-.5)* SG;
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