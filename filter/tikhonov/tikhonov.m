classdef tikhonov < filter
    %TIKHONOV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        U, T, Y0
        weights
        n               % Number of samples
        sz              % Size of the K or C matrix
        
        numGuesses      % number of filter hyperparameters guesses
        rng             % Parameter ranges map container
        currentParIdx   % Current parameter combination indexes map container
        currentPar      % Current parameter combination map container
        
    end
    
    methods
        
        function obj = tikhonov( K , Y , numSamples , numGuesses )
            
            if nargin > 3
                obj.init( K , Y , numSamples , numGuesses );
            elseif nargin > 2
                obj.init(  K , Y , numSamples );
            end
        end
        
        function init(obj , K , Y , numSamples , numGuesses)
                
            % Get number of samples
            obj.n = numSamples;
            
            % Get size of kernel/covariance matrix
            obj.sz = size(K,1);

            % Compute Hessenberger decomposition
            [obj.U, obj.T] = hess(K);
            obj.Y0 = obj.U' * Y;

            if( nargin == 5 )
                if numGuesses > 0
                    obj.numGuesses = numGuesses;
                else
                    obj.numGuesses = 1;
                end            
                obj.range();    % Compute range
                obj.currentParIdx = 0;
                obj.currentPar = [];
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

            lmin = max(min(lmax*smallnumber, eigmin), 200 * sqrt(eps));

            powers = linspace(0,1,obj.numGuesses);
            tmp = (lmin.*(lmax/lmin).^(powers)) / obj.n;        
            obj.rng = num2cell(tmp);
        end
        
        function compute(obj , filterPar )

            if( nargin > 1 )
                
                tmp1 = obj.T +  filterPar(1) * obj.n * eye(obj.sz);

                % Use sparse representation for tridiagonal matrix
                tmp1 = sparse(tmp1);

                % Invert
                %tic
                tmp2 = tmp1 \ obj.Y0;
                %toc

                obj.weights = obj.U * tmp2;

            % If any current value for any of the parameters is not available, abort.
            elseif (nargin == 1) && (isempty(obj.currentPar))
                error('Filter parameter(s) not explicitly specified, and some internal current parameters are not available available. Exiting...');
            else
                
                disp('Filter will be computed according to the internal current hyperparameter(s)');
                obj.currentPar

                tmp1 = obj.T +  obj.currentPar(1) * obj.n * eye(obj.sz);

                % Use sparse representation for tridiagonal matrix
                tmp1 = sparse(tmp1);

                % Invert
                %tic
                tmp2 = tmp1 \ obj.Y0;
                %toc

                obj.weights = obj.U * tmp2;

            end
        end
        
        % returns true if the next parameter combination is available and
        % updates the current parameter combination 'currentPar'
        function available = next(obj)

            % If any range for any of the parameters is not available, recompute all ranges.
            if cellfun(@isempty,obj.rng)
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