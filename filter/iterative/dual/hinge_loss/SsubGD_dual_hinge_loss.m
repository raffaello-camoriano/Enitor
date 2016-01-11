classdef SsubGD_dual_hinge_loss < filter
    %TIKHONOV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        
        X
        Y
        map                     % Kernel function in use
        mapPar                  % Kernel parameter(s)
        
        weights                 % Learned weights vector
        
        n                       % Number of samples
        sz                      % Size of the K or C matrix
        t                       % Number of outputs
        
        filterParGuesses        % Filter parameter guesses (range)
        numFilterParGuesses     % number of filter hyperparameters guesses
                
        eta                     % Step size
        theta
        
        currentParIdx           % Current parameter combination indexes map container
        currentPar              % Current parameter combination map container
        
        randOrd
    end
    
    methods
      
        function obj = SsubGD_dual_hinge_loss(map , mapPar , X , Y  , numSamples , varargin)

            obj.init(  map , mapPar , X , Y , numSamples , varargin );            
        end
        
        function init(obj , map , mapPar , X , Y , numSamples , varargin)
                 
            p = inputParser;
            
            %%%% Required parameters
            
            checkMap = @(x) true;
            checkMapPar = @(x) (x > 0);
            checkNumSamples = @(x) (x > 0);
            
            addRequired(p,'map',checkMap)
            addRequired(p,'mapPar',checkMapPar)
            addRequired(p,'X');
            addRequired(p,'Y');
            addRequired(p,'numSamples',checkNumSamples);
 
            %%%% Optional parameters
            % Optional parameter names:

            
            defaultEta = 1/4;
            checkEta = @(x) x > 0;

            defaultTheta = 1/2;
            checkTheta = @(x) (x <= 0 && x >= -1);

            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x) x >= 0;

            defaultFilterParGuesses = [];
            
            defaultInitialWeights = [];

            defaultVerbose = 0;
            checkVerbose = @(x) (x==0) || (x==1);

            addParameter(p,'eta',defaultEta,checkEta)
            addParameter(p,'theta',defaultTheta,checkTheta)
            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses,checkNumFilterParGuesses)
            addParameter(p,'filterParGuesses',defaultFilterParGuesses)
            addParameter(p,'initialWeights',defaultInitialWeights)
            addParameter(p,'verbose',defaultVerbose,checkVerbose)
            
            % Parse function inputs
            parse(p, map , mapPar , X , Y , numSamples , varargin{:}{:})
                        
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
            
            obj.eta = p.Results.eta;
            obj.theta = p.Results.theta;
            

            if isempty(p.Results.initialWeights)
                obj.weights = zeros(obj.n,1);
            else
                obj.weights = p.Results.initialWeights;
            end
            
            obj.map = p.Results.map;
            obj.mapPar = p.Results.mapPar;
            
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

            % Construct Kernel column
            argin = {};
            argin = [argin , 'mapParGuesses' , obj.mapPar];
            if ~isempty(obj.verbose)
                argin = [argin , 'verbose' , obj.verbose];
            end
            kernelLine = obj.map( obj.X ,  obj.X(randIdx , :) ,argin{:} );
            kernelLine.next();
            % Compute kernel according to current hyperparameters
            kernelLine.compute();
            
            Ypred = obj.weights' * kernelLine.K;
            
            if (Ypred * obj.Y(randIdx,:) <= 1)
                
                step = obj.eta * obj.currentPar^(-obj.theta);
                
%                 SG = - kernelLine.K * obj.Y(randIdx,:);
                SG = - obj.Y(randIdx,:);

                obj.weights = obj.weights - step * SG;
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