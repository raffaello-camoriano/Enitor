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
                
        etaGuesses                     % Step size
        thetaGuesses
        rng
        
        trainKernel
        initialWeights
        
        currentParIdx           % Current parameter combination indexes map container
        prevPar                 % Previous parameter
        currentPar              % Current parameter combination map container
                                % 1: Iteration
                                % 2: eta
                                % 3: theta
        randOrd
    end
    
    methods
      
        function obj = SsubGD_dual_hinge_loss(map , mapPar , X , Y  , numSamples , trainKernel , varargin)

            obj.init(  map , mapPar , X , Y , numSamples , trainKernel , varargin );            
        end
        
        function init(obj , map , mapPar , X , Y , numSamples , trainKernel , varargin)
                 
            p = inputParser;
            
            %%%% Required parameters
            
            checkTrainKernel = @(x) (size(trainKernel,1) > 0) && (size(trainKernel,2) > 0);
            checkMap = @(x) true;
            checkMapPar = @(x) (x > 0);
            checkNumSamples = @(x) (x > 0);
            
            addRequired(p,'map',checkMap)
            addRequired(p,'mapPar',checkMapPar)
            addRequired(p,'X');
            addRequired(p,'Y');
            addRequired(p,'numSamples',checkNumSamples);
            addRequired(p,'trainKernel',checkTrainKernel);

            %%%% Optional parameters
            % Optional parameter names:

            defaultEtaGuesses = 1/4;

            defaultThetaGuesses = 1/2;

            defaultNumFilterParGuesses = [];
            checkNumFilterParGuesses = @(x) x >= 0;

            defaultFilterParGuesses = [];
            
            defaultInitialWeights = [];

            defaultVerbose = 0;
            checkVerbose = @(x) (x==0) || (x==1);

            addParameter(p,'etaGuesses',defaultEtaGuesses)
            addParameter(p,'thetaGuesses',defaultThetaGuesses)
            addParameter(p,'numFilterParGuesses',defaultNumFilterParGuesses,checkNumFilterParGuesses)
            addParameter(p,'filterParGuesses',defaultFilterParGuesses)
            addParameter(p,'initialWeights',defaultInitialWeights)
            addParameter(p,'verbose',defaultVerbose,checkVerbose)
            
            % Parse function inputs
            parse(p, map , mapPar , X , Y , numSamples , trainKernel , varargin{:}{:})
                        
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
            
            obj.trainKernel = p.Results.trainKernel;

            % Check that only one between etaGuesses and thetaGuesses is an
            % array
            obj.etaGuesses = p.Results.etaGuesses;
            obj.thetaGuesses = p.Results.thetaGuesses;
            if ( numel(obj.etaGuesses) > 1 ) && ( numel(obj.thetaGuesses) > 1 )
                error('Only one between theta and eta can be crossvalidated. at least one must be fixed.');
            end

            % Set initial weights if specified.
            obj.initialWeights = p.Results.initialWeights;
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
            
            %% Compute range including eta xor theta guesses
            
            % crossvalidation of eta
            if numel(obj.etaGuesses) > 1
                
                [p,q] = meshgrid(obj.etaGuesses , obj.filterParGuesses);
                r = ones( numel(obj.etaGuesses) * numel(obj.filterParGuesses) , 1) * obj.thetaGuesses;
                tmp = [ q(:) p(:) r]';
            
            % crossvalidation of theta
            elseif numel(obj.thetaGuesses) > 1
                
                [p,q] = meshgrid(obj.thetaGuesses , obj.filterParGuesses);
                s = ones( numel(obj.thetaGuesses) * numel(obj.filterParGuesses) , 1) * obj.etaGuesses;
                tmp = [ q(:) s p(:) ]';
            % No crossvalidation
            else
                r = ones( numel(obj.thetaGuesses) * numel(obj.filterParGuesses) , 1) * obj.thetaGuesses;
                s = ones( numel(obj.thetaGuesses) * numel(obj.filterParGuesses) , 1) * obj.etaGuesses;
                tmp = [ obj.filterParGuesses' s r ]';
            end
            
            
            %% Generate all possible parameters combinations            
            
            obj.rng = num2cell(tmp , 1);
                        
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
            
            currIdx = obj.randOrd(mod(obj.currentPar(1)-1,obj.n) + 1);

            if isempty(obj.trainKernel)
                % Construct Kernel column according to current hyperparameters
                argin = {};
                argin = [argin , 'mapParGuesses' , obj.mapPar];
                if ~isempty(obj.verbose)
                    argin = [argin , 'verbose' , obj.verbose];
                end
                kernelLine = obj.map( obj.X ,  obj.X(currIdx , :) ,argin{:} );
                kernelLine.next();
                kernelLine.compute();
            
                % Compute prediction
                Ypred = obj.weights' * kernelLine.K;
            else
                % Precomputed kernel

                % Compute prediction
                Ypred = obj.weights' * obj.trainKernel(:,currIdx);
            end
            
            if (Ypred * obj.Y(currIdx,:) <= 1)
                
                % Compute step
                step = obj.currentPar(2) * obj.currentPar(1)^(-obj.currentPar(3));
                
                % Compute SubGradient
                SG = - obj.Y(currIdx,:);

                % Weights update iteration
                obj.weights(currIdx,:) = obj.weights(currIdx,:) - step * SG;
            end
        end
        
%         % returns true if the next parameter combination is available and
%         % updates the current parameter combination 'currentPar'
%         function available = next(obj)
% 
%             if isempty(obj.filterParGuesses)
%                 obj.range();
%             end
%             
%             available = false;
%             if length(obj.filterParGuesses) > obj.currentParIdx
%                 obj.currentParIdx = obj.currentParIdx + 1;
%                 obj.currentPar = obj.filterParGuesses(:,obj.currentParIdx);
%                 available = true;
%             end
%         end
        
        function resetPar(obj)
            
            obj.currentParIdx = 0;
            obj.currentPar = [];
            
        end
        
        function resetWeights(obj)
            
            % Set initial weights if specified.
            if isempty(obj.initialWeights)
                obj.weights = zeros(obj.n,1);
            else
                obj.weights = obj.initialWeights;
            end        
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            % If any range for any of the parameters is not available, recompute all ranges.
%             if cellfun(@isempty , obj.rng)
%                 obj.range();
%             end

            available = false;
            if length(obj.rng) > obj.currentParIdx
                obj.prevPar = obj.currentPar;
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.rng{obj.currentParIdx};
                available = true;
            end
        end                
    end
end
