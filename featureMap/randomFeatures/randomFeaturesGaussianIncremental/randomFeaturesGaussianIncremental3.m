classdef randomFeaturesGaussianIncremental3 < randomFeatures
    % 
    %
    % Input parameters:
    % TO DO.

    
    properties
                
        numRFParGuesses
        verbose
        
        numMapParRangeSamples   % Number of samples of X considered for estimating the maximum and minimum sigmas
        maxRank                 % Maximum rank of the kernel approximation
        minRank                 % Minimum rank of the kernel approximation
        
        filterPar        % Filter parameter guesses
        
        mapParGuesses           % mapping parameter guesses
%         numMapParGuesses        % Number of mapping parameter guesses
        
        kernelType      % Type of approximated kernel
        SqDistMat       % Squared distances matrix
        Y               % Training outputs
        Xs              % Sampled points
        
        s               % Number of new columns
        ntr             % Total number of training samples
        t
        
        prevPar
        
        % Variables for computeSqDistMat efficiency
        X1
        X2
        Sx1
        Sx2
        
        A
        Aty
        R
        alpha        
        
    end
    
    methods
        
        function o = randomFeaturesGaussianIncremental3( X , Y , ntr , varargin)
            o.init( X , Y , ntr , varargin);
        end

        function o = init(o , X , Y , ntr , varargin)
            
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
            % numRFParGuesses , maxRank , numMapParGuesses , mapParGuesses , filterPar , numMapParRangeSamples  , verbose
            
            % numRFParGuesses       % Cardinality of number of samples for Nystrom approximation parameter guesses
            defaultNumRFParGuesses = [];
            checkNumRFParGuesses = @(x) x > 0 ;            
            addParameter(p,'numRFParGuesses',defaultNumRFParGuesses,checkNumRFParGuesses);                    
            
            % minRank        % Minimum rank of the RF approximation
            defaultMinRank = [];
            checkMinRank = @(x) x > 0 ;            
            addParameter(p,'minRank',defaultMinRank,checkMinRank);        
            
            % maxRank        % Maximum rank of the RF approximation
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
            
            % filterPar       % Filter parameter guesses
            defaultFilterParGuesses = [];
            checkFilterParGuesses = @(x) size(x,1) > 0 && size(x,2) > 0 ;            
            addParameter(p,'filterPar',defaultFilterParGuesses,checkFilterParGuesses);                    
            
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
            for idx = 1:numel(fields)
                o.(fields{idx}) = p.Results.(fields{idx});
            end
            
            % Joint parameters parsing
            if o.minRank > o.maxRank
                error('The specified minimum rank of the kernel approximation is larger than the maximum one.');
            end 
            
            if isempty(o.mapParGuesses) && isempty(o.numMapParGuesses)
                error('either mapParGuesses or numMapParGuesses must be specified');
            end    
            
            if (~isempty(o.mapParGuesses)) && (~isempty(o.numMapParGuesses)) && (size(o.mapParGuesses,2) ~= o.numMapParGuesses)
                error('The size of mapParGuesses and numMapParGuesses are different');
            end
            
            if ~isempty(o.mapParGuesses) && isempty(o.numMapParGuesses)
                o.numMapParGuesses = size(o.mapParGuesses,2);
            end
            
            if size(X,1) ~= size(Y,1)
                error('X and Y have incompatible sizes');
            end

            o.d = size(X , 2);
            o.t = size(Y , 2);            
            
            display('Kernel used by randomFeaturesGaussianIncremental is set to @gaussianKernel');
            o.kernelType = @gaussianKernel;

            % Conditional range computation
%             if isempty(o.mapParGuesses)
            if o.verbose == 1
                display('Computing range');
            end
            o.range();    % Compute range
%             end
            o.currentParIdx = 0;
            o.currentPar = [];
        end

        function mappedSample = map(o , inputSample , partialRange)
            
            % [cos sin] mapping
%             V = inputSample * o.omega;
%             mappedSample = sqrt( 2 / o.currentPar(1) ) * [cos(V) , sin(V)];
            
            if isempty(partialRange)
%                 Full cos(wx+b) mapping
                V =  inputSample * o.omega + repmat(o.b , size(inputSample,1) , 1);            
                mappedSample = cos(V);
            else
                % Partial cos(wx+b) mapping
                V =  inputSample * o.omega(:,partialRange) + repmat(o.b(partialRange) , size(inputSample,1) , 1);            
                mappedSample = cos(V);
            end
        end
        
        function o = range(o)
                        
            % Range of the number of Random Fourier Features
            tmpNumRF = round(linspace(o.minRank, o.maxRank , o.numRFParGuesses));   
           
            if isempty(o.mapParGuesses)
                % Compute max and min sigma guesses

                % Extract an even number of samples without replacement                
                nRows = size(o.X,1); % number of rows
                nSample = o.numMapParRangeSamples - mod(o.numMapParRangeSamples,2); % number of samples
                
                rndIDX = [];
                while length(rndIDX) < nSample
                    rndIDX = [rndIDX , randperm(nRows , min( [ nSample , nRows , nSample - length(rndIDX) ] ) ) ];
                end
                
                samp = o.X(rndIDX, :);   
                
                % Compute squared distances  vector (D)
                numDistMeas = floor(o.numMapParRangeSamples/2); % Number of distance measurements
                D = zeros(1 , numDistMeas);
                for i = 1:numDistMeas
                    D(i) = sum((samp(2*i-1,:) - samp(2*i,:)).^2);
                end
                D = sort(D);

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

                tmpMapPar = linspace(minGuess, maxGuess , o.numMapParGuesses);
            else
                tmpMapPar = o.mapParGuesses;
            end
            
            % Generate all possible parameters combinations
            [p,q] = meshgrid(tmpMapPar, tmpNumRF);
            tmp = [q(:) p(:)]';
            o.rng = tmp;

        end
        
        function compute(o , mapPar)
            
            if( nargin > 1 )
                
%                 if(o.verbose == 1)
%                     disp('Mapping will be computed according to the provided hyperparameter(s)');
%                     mapPar
%                 end
                chosenPar = mapPar;
            elseif (nargin == 1) && (isempty(o.currentPar))
                
                % If any current value for any of the parameters is not available, abort.
                error('Mapping parameter(s) not explicitly specified, and some internal current parameters are not available available. Exiting...');
            else
%                 if(o.verbose == 1)
%                     disp('Mapping will be computed according to the current internal hyperparameter(s)');
%                     o.currentPar
%                 end
                chosenPar = o.currentPar;
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Incremental Update Rule %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%

            if (isempty(o.prevPar) && o.currentParIdx == 1) || ...
                    (~isempty(o.prevPar) && o.currentPar(1) < o.prevPar(1))
                
                %%% Initialization (i = 1)
                
                % Preallocate matrices
                
                o.A = zeros(o.ntr , o.maxRank);
                o.Aty = zeros(o.maxRank , o.t);
                
                o.R = deal(zeros(o.maxRank));
                o.alpha = deal(zeros(o.maxRank,size(o.Y,2)));
                
                o.s = chosenPar(1);
                [o.omega , o.b] = o.generateProj(chosenPar);

                % Set Xs
                o.Xs = o.map(o.X , []);

                o.A(:,1:chosenPar(1)) = o.Xs;
                o.Aty(1:chosenPar(1),:) = o.A(:,1:chosenPar(1))' * o.Y;

                    
                % Cholesky factor R
                o.R(1:chosenPar(1),1:chosenPar(1)) = ...
                    chol(full(o.A(:,1:chosenPar(1))' * o.A(:,1:chosenPar(1)) ) + ...
                    o.ntr *  o.filterPar * eye(chosenPar(1)) );

                % alpha
                o.alpha = 	o.R(1:chosenPar(1),1:chosenPar(1)) \ ...
                    ( o.R(1:chosenPar(1),1:chosenPar(1))' \ ...
                    ( o.Aty(1:chosenPar(1),:) ) );
                
            elseif o.prevPar(1) ~= chosenPar(1)
               
                %%% Generic i-th incremental update step
               
                sampledPoints = (o.prevPar(1) + 1):chosenPar(1);                
                o.s = chosenPar(1) - o.prevPar(1);  % Number of new columns
     
                % Generate new random projections
                [newOmega , newB] = o.generateProj([o.s ; chosenPar(2)]);
                o.omega = [o.omega , newOmega];
                o.b = [o.b , newB];
                
                % Compute a
                a = o.map(o.X , sampledPoints) ;
                
                % Update Xs_(t+1)
%                 o.Xs = [o.Xs a];    
                

                % Compute c, gamma
                c = o.A(:,1:o.prevPar(1))' * a;
%                     gamma = a' * a + (o.ntr * o.filterPar + 1e-8) * eye(numel(sampledPoints));
                gamma = a' * a + (o.ntr * o.filterPar) * eye(numel(sampledPoints));


                % Update A, Aty
                o.A( : , (o.prevPar(1)+1) : chosenPar(1) ) = a ;
                o.Aty((o.prevPar(1)+1) : chosenPar(1) , : ) = a' * o.Y ;

                % Compute u, v
%                     u = [ c / ( 1 + sqrt( 1 + gamma) ) ; ...
%                                     sqrt( 1 + gamma)               ];
%                     
%                     v = [ c / ( 1 + sqrt( 1 + gamma) ) ; ...
%                                     -1               ];
% 
% 
%                     % Update R
%                     
%                     o.R{i}(1:chosenPar(1),1:chosenPar(1)) = ...
%                         cholupdatek( o.R{i}(1:chosenPar(1),1:chosenPar(1)) , u , '+');
% 
% %                     try
%                     o.R{i}(1:chosenPar(1),1:chosenPar(1)) = ...
%                         cholupdatek(o.R{i}(1:chosenPar(1),1:chosenPar(1)) , v , '-');

                u = [ 2 * c / ( (sqrt( 3 ) - 1 ) * sqrt(gamma)) ; ...
                                sqrt( 0.75 * gamma)               ];

                v = [ - 2 * c / ( (sqrt( 3 ) - 1 ) * sqrt(gamma)) ; ...
                                sqrt( gamma / 4)               ];


                % Update R

                o.R(1:chosenPar(1),1:chosenPar(1)) = ...
                    cholupdatek( o.R(1:chosenPar(1),1:chosenPar(1)) , u , '+');

                o.R(1:chosenPar(1),1:chosenPar(1)) = ...
                    cholupdatek(o.R(1:chosenPar(1),1:chosenPar(1)) , v , '+');

                % Recompute alpha
                o.alpha = o.R(1:chosenPar(1),1:chosenPar(1)) \ ...
                    ( o.R(1:chosenPar(1),1:chosenPar(1))' \ ...
                    ( o.Aty(1:chosenPar(1),:) ) );
            end
        end
        
        function resetPar(o)
            
            o.currentParIdx = 0;
            o.currentPar = [];
        
        end
        
        function [omega , b] = generateProj(o , mapPar)

            omega =  randn(o.d, mapPar(1)) / mapPar(2);
            b =  rand(1,mapPar(1))* 2 * pi;
        end

        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(o)

            % If any range for any of the parameters is not available, recompute all ranges.
%             if cellfun(@isempty,o.mapParGuesses)
%                 o.range();
%             end
            if isempty(o.rng)
                o.range();
            end
            
            available = false;
%             if length(o.mapParGuesses) > o.currentParIdx
%                 o.currentParIdx = o.currentParIdx + 1;
%                 o.currentPar = o.mapParGuesses{o.currentParIdx};
%                 available = true;
%             end
            if size(o.rng,2) > o.currentParIdx
                o.prevPar = o.currentPar;
                o.currentParIdx = o.currentParIdx + 1;
                o.currentPar = o.rng(:,o.currentParIdx);
                available = true;
            end
        end
    end
end
