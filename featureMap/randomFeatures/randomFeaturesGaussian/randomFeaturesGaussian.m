classdef randomFeaturesGaussian < randomFeatures
    %RANDOMFEATURES Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        numKerParRangeSamples   % Number of samples of X considered for estimating the maximum and minimum sigmas
    end
    
    methods
        % Constructor
        function obj = randomFeaturesGaussian( X , numMapParGuesses , numKerParRangeSamples )
            
            obj.init( X , numMapParGuesses , numKerParRangeSamples );
            
        end
        
        % Initialization function
        function obj = init(obj , X , numMapParGuesses , numKerParRangeSamples )
            
            obj.X = X;
            obj.numMapParGuesses = numMapParGuesses;
            obj.numKerParRangeSamples = numKerParRangeSamples;
            obj.d = size(X , 2);     
            
            % Compute range
            obj.range();
            obj.currentParIdx = 0;
            obj.currentPar = [];
        end

        % WARNING: Modify sampling strategy
        % Computes the squared distance matrix SqDistMat based on X1, X2
        function SqDistMat = computeSqDistMat(obj , X1 , X2)
            
            p = size(X1,1);
            q = size(X2,1);
            
            Sx1 = sum( X1.*X1 , 2);
            Sx2 = sum( X2.*X2 , 2)';
            Sx1x2 = X1 * X2';
            
            SqDistMat = repmat(Sx1 , 1 , p) -2 * Sx1x2 + repmat(Sx2 , q , 1);
        
        end
        
        function mappedSample = map(obj , inputSample)
            V = inputSample * obj.omega;
            mappedSample = sqrt( 2 / obj.currentPar(1) ) * [cos(V) , sin(V)];
            
%             mappedSample = real(exp(1i*inputSample * obj.omega));
        end
        
        function obj = range(obj)
            %% Range of the number of Random Fourier Features
            
            tmpNumRF = linspace(obj.d, 2000 , obj.numMapParGuesses);   
            
            %% Approximated kernel parameter range
            
            % Compute max and min sigma guesses (same strategy used by
            % GURLS)

            %Compute partial square distances matrix
            SqDistMat = obj.computeSqDistMat( obj.X(1:obj.numKerParRangeSamples,:) , obj.X(1:obj.numKerParRangeSamples,:) );
            
            D = sort(SqDistMat(tril(true(obj.numKerParRangeSamples),-1)));
            firstPercentile = round(0.01*numel(D)+0.5);
            minGuess = sqrt(D(firstPercentile));
            maxGuess = sqrt(max(max(SqDistMat)));

            if minGuess <= 0
                minGuess = eps;
            end
            if maxGuess <= 0
                maxGuess = eps;
            end	
            
            tmpKerPar = linspace(minGuess, maxGuess , obj.numMapParGuesses);
            
            %% Generate all possible parameters combinatins            
            
            [p,q] = meshgrid(tmpNumRF, tmpKerPar);
            tmp = [p(:) q(:)]';
            
            obj.rng = num2cell(tmp , 1);
            
        end
        
        function compute(obj , mapPar)
            
            if( nargin > 1 )
                
                disp('Mapping will be computed according to the provided hyperparameter(s)');
                mapPar
                chosenPar = mapPar;
                
            elseif (nargin == 1) && (isempty(obj.currentPar))
                
                % If any current value for any of the parameters is not available, abort.
                error('Mapping parameter(s) not explicitly specified, and some internal current parameters are not available available. Exiting...');
            
            else
                
                disp('Mapping will be computed according to the internal current hyperparameter(s)');
                obj.currentPar
                chosenPar = obj.currentPar;
                
            end
            
            obj.generateProj(chosenPar);
            
%             obj.Xrf = 	 real(exp(1i*obj.X * obj.omega));

            V =  obj.X * obj.omega;
            obj.Xrf = sqrt( 2 / chosenPar(1) ) * [cos(V) , sin(V)];
        end        
        
        function obj = generateProj(obj , mapPar)
            
            % TODO: Vary sigma parameter!!!
            obj.omega =  mapPar(2) * randn(obj.d, mapPar(1));
            %obj.omega = sqrt(2) * randn(obj.d, mapPar(1));
            %obj.omega = mapPar(2) * randn(obj.d, mapPar(1));
            %obj.omega = mapPar(2) * (2 * pi)^(-mapPar(1)/2) * randn(obj.d, mapPar(1));

        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            % If any range for any of the parameters is not available, recompute all ranges.
            if cellfun(@isempty , obj.rng)
                obj.range();
            end

            available = false;
            if length(obj.rng) > obj.currentParIdx
                obj.currentParIdx = obj.currentParIdx + 1;
                obj.currentPar = obj.rng{obj.currentParIdx};
                available = true;
            end
        end        
    end
end
