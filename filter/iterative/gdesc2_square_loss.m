classdef gdesc2_square_loss < filter
    %TIKHONOV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        
        % Parameters used in the classical formulation
        X 						% Training pts
        Y						% Training labels
        
        weights                 % Learned weights vector
        n                       % Number of samples
        d                       % dimensionality of X
        t                       % Number of outputs
        
        filterParGuesses        % Filter parameter guesses (range)
        numFilterParGuesses     % number of filter hyperparameters guesses
        
        kappa                   % Used to compute the optimal step size gamma at each step t
        gamma                   % Step size
        
        currentParIdx           % Current parameter combination indexes map container
        currentPar              % Current parameter combination map container
    end
    
    methods
      
        function obj = gdesc2_square_loss( X , Y , numSamples , varargin)

            obj.init(  X , Y , numSamples , varargin );            
        end
        
        function init(obj , X , Y , numSamples , varargin)
                 
            p = inputParser;
            
            %%%% Required parameters
            
            checkX = @(x) size(x,1) > 0 && size(x,2) > 0;
            checkNumSamples = @(x) (x > 0);
            
            addRequired(p,'X',checkX);
            addRequired(p,'Y');
            addRequired(p,'numSamples',checkNumSamples);
            
            %%%% Optional parameters
            % Optional parameter names:

%             defaultGamma = 2/numSamples;
%             eigmax = eigs(K,[],1);
%             sv = svd(K);
%             svmax = sv(1);
%             defaultGamma = 2/eigmax^2;
%             defaultGamma = 1/(2*numSamples);
%            defaultGamma = 1/numSamples;
%             defaultGamma = 2/numSamples;
% 			L = eigs(X'*X,1);
            L = max(abs(diag(X*X')));
% 			L = eigs(X*X',1)/sqrt(numSamples);
			defaultGamma = 1/(L*numSamples);
            checkGamma = @(x) x > 0;

            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x) x >= 0;

            defaultFilterParGuesses = [];
%             checkFixedFilterParGuesses = @(x) x >= 0;
            
            defaultVerbose = 0;
            checkVerbose = @(x) (x==0) || (x==1);

            addParameter(p,'gamma',defaultGamma,checkGamma)
            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses,checkNumFilterParGuesses)
            addParameter(p,'filterParGuesses',defaultFilterParGuesses)
            addParameter(p,'verbose',defaultVerbose,checkVerbose)
            
            % Parse function inputs
            parse(p, X , Y , numSamples , varargin{:}{:})
                        
            
            % Get number of samples
            obj.n = p.Results.numSamples;

            % Store full kernel/covariance matrix
            obj.X = p.Results.X;
            obj.Y = p.Results.Y;

            % Get size of kernel/covariance matrix
            obj.d = size(obj.X,2);
            
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
            
            obj.weights = obj.weights + obj.gamma * obj.X(mod(obj.currentPar-1,obj.n)+1,:)' * (obj.Y(mod(obj.currentPar-1,obj.n)+1,:) - obj.X(mod(obj.currentPar-1,obj.n)+1,:)  * obj.weights);
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
