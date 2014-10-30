classdef tikhonov < filter
    %TIKHONOV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        U, T, Y0
        coeffs
        n % Number of samples
        
        numGuesses      % number of filter hyperparameters guesses
        rng             % Parameter ranges map container
        currentParIdx   % Current parameter combination indexes map container
        currentPar      % Current parameter combination map container
    end
    
    methods
        
        function obj = tikhonov( K , Y , numGuesses , filterPar)
            
            if  nargin > 0
                if  nargin > 3
                    obj.init( K , Y , numGuesses, filterPar );
                elseif nargin > 2
                    obj.init( K , Y , numGuesses );
                elseif nargin > 1
                    obj.init( K , Y );
                end
            end
        end
        
        function init(obj , K , Y , numGuesses, filterPar)
            
            % Get number of samples
            obj.n = size(K,1);

            % Compute Hessenberger decomposition
            [obj.U, obj.T] = hess(K);
            obj.Y0 = obj.U'*Y;
            
            if( nargin == 4 )
                if numGuesses > 0
                    obj.numGuesses = numGuesses;
                else
                    obj.numGuesses = 1;
                end            

                % Initialize range map
                rangeKeySet = {'lambda'};
                rangeValueSet = cell(size(rangeKeySet,1));
                rangeValueSet{:,:} = zeros(obj.numGuesses,1);
                obj.rng = containers.Map(rangeKeySet,rangeValueSet);

                % Initialize current parameter combination indexes map
                currentParIdxKeySet = rangeKeySet;
                currentParIdxValueSet = cell(size(currentParIdxKeySet,1));
                currentParIdxValueSet{:,:} = 0;
                obj.currentParIdx = containers.Map(currentParIdxKeySet,currentParIdxValueSet);
                
                % Initialize current parameter combination map
                currentParKeySet = rangeKeySet;
                currentParValueSet = cell(size(currentParIdxKeySet,1));
                obj.currentPar = containers.Map(currentParKeySet,currentParValueSet);
            
            elseif  nargin > 4
                obj.compute(filterPar);
            end            
        end
        
        function obj = range(obj)
            
            % TODO: @Ale: Smart computation of eigmin and eigmax starting from the
            % tridiagonal matrix U
            
            % ...
            
            % GURLS code below, set 'eigmax' and 'eigmin' variables
            
            
            %===================================================
            % DEBUG: Dumb computation of min and max eigenvalues
            
%             % Reconstruct kernel matrix
%             K = obj.U * obj.T * obj.U';
%             
%             % Perform SVD of K
%             e = eig(K);
%             
%             % Grab min and max eigs
%             eigmax = max(e);
%             eigmin = min(e);
            
            % DEBUG: fixed minimum and maximum eigenvalues
            eigmax = 1;
            eigmin = 10e-7;
            %===================================================
            
            % maximum lambda
            lmax = eigmax;
            
            smallnumber = 1e-8;

            % just in case, when r = min(n,d) and r x r has some zero eigenvalues
            % we take a max; 200*sqrt(eps) is the legacy number used in the previous
            % code, so i am just continuing it.

            lmin = max(min(lmax*smallnumber, eigmin), 200*sqrt(eps));

            powers = linspace(0,1,obj.numGuesses);
            obj.rng('lambda') = (lmin.*(lmax/lmin).^(powers))/obj.n;            
        end
        
        function compute(obj , filterPar)

            if( nargin > 1 )
                obj.coeffs = obj.U * (( obj.T + filterPar('lambda') * obj.n * eye(obj.n)) \ obj.Y0);
            
            % If any current value for any of the parameters is not available, abort.
            elseif (nargin == 1) && (sum(cellfun(@isempty,values(obj.currentPar))) > 0)
                error('Filter parameter(s) not explicitly specified, and some internal current parameters are not available available. Exiting...');
            else
                disp('Filter will be computed according to the internal current hyperparameter(s)');
                obj.currentPar('lambda')
                obj.coeffs = obj.U * (( obj.T + obj.currentPar('lambda') * obj.n * eye(obj.n)) \ obj.Y0);
            end
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)
            
            % If any range for any of the parameters is not available, recompute all ranges.
            if sum(cellfun(@isempty,values(obj.rng))) > 0
                obj.range
            end
                        
            available = false;
            for key = keys(obj.rng)
                keyStr = key{1};
                if length(obj.rng(keyStr)) >= obj.currentParIdx(keyStr) + 1
                    obj.currentParIdx(keyStr) = obj.currentParIdx(keyStr) + 1;
                    
                    tmp = obj.rng(keyStr);
                    obj.currentPar(keyStr) = tmp(obj.currentParIdx(keyStr));
                    
                    available = true;
                end
            end
        end        
    end
end