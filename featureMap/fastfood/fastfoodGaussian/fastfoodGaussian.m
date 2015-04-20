classdef fastfoodGaussian < fastfood
    %RANDOMFEATURES Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numMapParRangeSamples    % Number of samples of X considered for estimating the maximum and minimum sigmas
        numRFParGuesses
        maxRank     % Maximum number of random features to be used
        kernelType
        
        mapParGuesses
        
        ntr     % Number of training points
        
        use_spiral
        
        verbose
        
        Y

    end
    
    methods
        % Constructor
        function obj = fastfoodGaussian(  X , Y , ntr , varargin)
            
            obj.init(  X , Y , ntr , varargin)
        end
        
        % Initialization function
        function obj = init(obj ,  X , Y , ntr , varargin)
            
            p = inputParser;
            
            %%%% Required parameters
            % X , Y , ntr

            checkX = @(x) size(x,1) > 0 && size(x,2) > 0;
            checkY = @(x) size(x,1) > 0 && size(x,2) > 0;
            checkNtr = @(x) x > 0;
            
            addRequired(p,'X',checkX);
            addRequired(p,'Y',checkY);
            addRequired(p,'ntr',checkNtr);
            
 
            %%%% Optional parameters
            % Optional parameter names:
            % numRFParGuesses , maxRank , numMapParGuesses , mapParGuesses , filterParGuesses , numMapParRangeSamples  , verbose
            
            % numRFParGuesses       % Cardinality of number of samples for Nystrom approximation parameter guesses
            defaultNumRFParGuesses = [];
            checkNumRFParGuesses = @(x) x > 0 ;            
            addParameter(p,'numRFParGuesses',defaultNumRFParGuesses,checkNumRFParGuesses);                    
            
            % maxRank        % Maximum rank of the Nystrom approximation
            defaultMaxRank = [];
            checkMaxRank = @(x) x > 0 ;            
            addParameter(p,'maxRank',defaultMaxRank,checkMaxRank);        
            
            % numMapParGuesses        % Number of map parameter guesses
            defaultNumMapParGuesses = [];
            checkNumMapParGuesses = @(x) x > 0 ;            
            addParameter(p,'numMapParGuesses',defaultNumMapParGuesses,checkNumMapParGuesses);        
            
            % mapParGuesses        % Map parameter guesses
            defaultMapParGuesses = [];
            checkMapParGuesses = @(x) size(x,1) > 0 && size(x,2) > 0 ;      
            addParameter(p,'mapParGuesses',defaultMapParGuesses,checkMapParGuesses);        
            
            % numMapParRangeSamples        % Number of map parameter guesses
            defaultNumMapParRangeSamples = [];
            checkNumMapParRangeSamples = @(x) x > 0 ;            
            addParameter(p,'numMapParRangeSamples',defaultNumMapParRangeSamples,checkNumMapParRangeSamples);            

            % verbose             % 1: verbose; 0: silent      
            defaultVerbose = 0;
            checkVerbose = @(x) (x == 0) || (x == 1) ;            
            addParameter(p,'verbose',defaultVerbose,checkVerbose);
            
            % Parse function inputs
            if isempty(varargin{:})
                parse(p, X , Y , ntr)
            else
                parse(p, X , Y , ntr , varargin{:}{:})
            end
            
            % Assign parsed parameters to object properties
            fields = fieldnames(p.Results);
%             fieldsToIgnore = {'X1','X2'};
%             fields = setdiff(fields, fieldsToIgnore);
            for idx = 1:numel(fields)
                obj.(fields{idx}) = p.Results.(fields{idx});
            end
            
            % Joint parameters parsing
            if isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                error('either mapParGuesses or numMapParGuesses must be specified');
            end    
            
            if (~isempty(obj.mapParGuesses)) && (~isempty(obj.numMapParGuesses)) && (size(obj.mapParGuesses,2) ~= obj.numMapParGuesses)
                error('The size of mapParGuesses and numMapParGuesses are different');
            end
            
            if ~isempty(obj.mapParGuesses) && isempty(obj.numMapParGuesses)
                obj.numMapParGuesses = size(obj.mapParGuesses,2);
            end
            
            if size(X,1) ~= size(Y,1)
                error('X and Y have incompatible sizes');
            end
            
            obj.d = size(X , 2);
            
            warning('Kernel used by randomFeaturesGaussianIncremental is set to @gaussianKernel');
            obj.kernelType = @gaussianKernel;
            
            % Compute Fastfood parameters
            obj.para = FastfoodPara(obj.maxRank, obj.d);
            
            obj.use_spiral = 0;

            % Compute range
            obj.range();
            obj.currentParIdx = 0;
            obj.currentPar = [];
        end
        
        function mappedSample = map(obj , inputSample)
            
            mappedSample = transpose(FastfoodForKernel(inputSample', obj.para, obj.currentPar(1), obj.use_spiral));
      
        end
        
        function obj = range(obj)
                        
            % Range of the number of Random Fourier Features
            tmpNumRF = round(linspace(1, obj.maxRank , obj.numRFParGuesses));   
           
            if isempty(obj.mapParGuesses)
                % Compute max and min sigma guesses

                % Extract an even number of samples without replacement                

                % WARNING: not compatible with versions older than 2014
                %samp = datasample( obj.X(:,:) , obj.numKerParRangeSamples - mod(obj.numKerParRangeSamples,2) , 'Replace', false);

                % WARNING: Alternative to datasample below
                nRows = size(obj.X,1); % number of rows
                nSample = obj.numMapParRangeSamples - mod(obj.numMapParRangeSamples,2); % number of samples
                rndIDX = randperm(nRows); 
                samp = obj.X(rndIDX(1:nSample), :);   

                % Compute squared distances  vector (D)
                numDistMeas = floor(obj.numMapParRangeSamples/2); % Number of distance measurements
                D = zeros(1 , numDistMeas);
                for i = 1:numDistMeas
                    D(i) = sum((samp(2*i-1,:) - samp(2*i,:)).^2);
                end
                D = sort(D);

                firstPercentile = round(0.01 * numel(D) + 0.5);
                minGuess = sqrt( D(firstPercentile));
                maxGuess = sqrt( max(D) );

                if minGuess <= 0
                    minGuess = eps;
                end
                if maxGuess <= 0
                    maxGuess = eps;
                end	

                tmpMapPar = linspace(minGuess, maxGuess , obj.numMapParGuesses);
            else
                tmpMapPar = obj.mapParGuesses;
            end

            % Generate all possible parameters combinations
            [p,q] = meshgrid(tmpNumRF, tmpMapPar);
            tmp = [p(:) q(:)]';
%             obj.rng = num2cell(tmp , 1);
            obj.rng = tmp;

        end
        
        function compute(obj , mapPar)
            
            if( nargin > 1 )
                
                if(obj.verbose == 1)
                    disp('Mapping will be computed according to the provided hyperparameter(s)');
                    mapPar
                end
                chosenPar = mapPar;
            elseif (nargin == 1) && (isempty(obj.currentPar))
                
                % If any current value for any of the parameters is not available, abort.
                error('Mapping parameter(s) not explicitly specified, and some internal current parameters are not available available. Exiting...');
            else
                if(obj.verbose == 1)
                    disp('Mapping will be computed according to the current internal hyperparameter(s)');
                    obj.currentPar
                end
                chosenPar = obj.currentPar;
            end
            
            obj.Xrf = FastfoodForKernel(obj.X', obj.para, chosenPar(1), obj.use_spiral);
            obj.Xrf = obj.Xrf';
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            if isempty(obj.rng)
                obj.range();
            end
            
            available = false;
            
            if length(obj.rng) > obj.currentParIdx
                obj.prevPar = obj.currentPar;
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.rng(:,obj.currentParIdx);
                available = true;
            end
        end    
    end
end
