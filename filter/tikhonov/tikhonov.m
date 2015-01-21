classdef tikhonov < filter
    %TIKHONOV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        
        M               % sz * sz matrix to be multiplied by lambda. M = speye(sz) by default
        sparseFlag          % Flag insicating is the sparse (1) or classical (0) formulation is being used
        
        % Parameters used in the classical formulation
        K
        Y
        
        % Parameters used in the sparse formulation
        U
        Y0
        T
        
        weights         % Learned weights vector
        n               % Number of samples
        sz              % Size of the K or C matrix
        
        fixedFilterPar  % Fixed filter parameter
        
        numGuesses      % number of filter hyperparameters guesses
        rng             % Parameter ranges map container
        currentParIdx   % Current parameter combination indexes map container
        currentPar      % Current parameter combination map container
    end
    
    methods
        
        function obj = tikhonov( K , Y , numSamples , numGuesses , M , fixedFilterPar)
            
            if nargin > 5
                obj.init( K , Y , numSamples , numGuesses , M , fixedFilterPar);     
            elseif nargin > 4
                obj.init( K , Y , numSamples , numGuesses , M );            
            elseif nargin > 3
                obj.init( K , Y , numSamples , numGuesses );
            elseif nargin > 2
                obj.init(  K , Y , numSamples );
            end
        end
        
        function init(obj , K , Y , numSamples , numGuesses , M, fixedFilterPar)
                
            % Check dimension of K
            if size(K,1) ~= size(K,2)
                error('K must be squared.');
            end
    
            % Check dimension of Y
            if size(K,1) ~= numSamples
                error('The number of rows of Y must be == numSamples.');
            end
            
            % Get number of samples
            obj.n = numSamples;
            
            % Get size of kernel/covariance matrix
            obj.sz = size(K,1);

            if nargin >= 6 && ~isempty(M)
                
                % Check M's size
                if size(M,1) ~= size(M,2) || size(M,1) ~= size(K,1)
                    error ('M must be squared, and its size must be the same as size(K)');
                end
                
                % Set sparsity flag
                if sum(sum(M ~= eye(size(M)))) > 0
                    obj.sparseFlag = 0;
                    obj.M = M;                
                else
                    obj.sparseFlag = 1;
                    obj.M = speye(size(M));
                end

            else
                obj.sparseFlag = 1;
                obj.M = speye(obj.sz);
            end
            
            if obj.sparseFlag == 1
                % Compute Hessenberger decomposition
                [obj.U, T] = hess(full(K));
                obj.T = sparse(T);  % Store as sparse matrix (it is tridiagonal)
                T = [];
                obj.Y0 = obj.U' * Y;
            else
                % Store full kernel/covariance matrix
                obj.K = K;
                obj.Y = Y;
            end
            
            % Compute hyperparameter(s) range
            if( nargin >= 5  && nargin <7)
                if numGuesses > 0
                    obj.numGuesses = numGuesses;
                else
                    obj.numGuesses = 1;
                end            
                obj.range();    % Compute range
                obj.currentParIdx = 0;
                obj.currentPar = [];
            end          
            
            % Compute hyperparameter(s) range
            if nargin >= 7  && ~isempty(fixedFilterPar)

                obj.numGuesses = 1;
                obj.fixedFilterPar = fixedFilterPar;
                
                obj.range();    % Compute range
                obj.currentParIdx = 0;
                obj.currentPar = [];
            end                
            
        end
        
        function obj = range(obj)
            
            if isempty(obj.fixedFilterPar)
            
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

    %             e = eigs(obj.T,1);


    %             eigmax = norm(obj.T);
                % WARNING: Error using norm
                % Sparse norm(S,2) is not available.

                % DEBUG: fixed minimum and maximum eigenvalues
                %eigmax = 100;
                %eigmin = 10e-7;


                %===================================================

                if obj.sparseFlag == 1
                    eigmax = eigs(obj.T,1);
                    opts.issym = 1;
    %                 eigmin = eigs(obj.K,1,'sm', opts);                
                    eigmin = eigs(obj.T,1,10e-7, opts);
                else
                    eigmax = eigs(obj.K,1);
                    opts.issym = 1;
    %                 eigmin = eigs(obj.K,1,'sm', opts);
                    eigmin = eigs(obj.K,1,10e-7, opts);
                end    

                % maximum lambda
                lmax = eigmax;
                smallnumber = 1e-8;

                % just in case, when r = min(n,d) and r x r has some zero eigenvalues
                % we take a max; 200*sqrt(eps) is the legacy number used in the previous
                % code, so i am just continuing it.

                lmin = max(min(lmax*smallnumber, eigmin), 200 * sqrt(eps));

                powers = linspace(1,0,obj.numGuesses);
                tmp = (lmin.*(lmax/lmin).^(powers)) / obj.n;
                
            else
                tmp = obj.fixedFilterPar;
            end
            
            obj.rng = num2cell(tmp);
        end
        
        function compute(obj , filterPar )

            if( nargin > 1 )
                disp('Filter will be computed according to the given hyperparameter(s)');
                selectedPar = filterPar
            elseif (nargin == 1) && (isempty(obj.currentPar))
                % If any current value for any of the parameters is not available, abort.
                error('Filter parameter(s) not explicitly specified, and some internal current parameters are not available available. Exiting...');
            else
                disp('Filter will be computed according to the internal current hyperparameter(s)');
                selectedPar = obj.currentPar
            end
            
            if obj.sparseFlag == 1
                % Compute weights according to the filter parameter(s)
                tmp1 = obj.T + selectedPar(1) * obj.n * obj.M;

                % Invert
                %tic
                tmp2 = tmp1 \ obj.Y0;
                %toc

                obj.weights = obj.U * tmp2;
            else
                obj.weights = (obj.K + selectedPar(1) * obj.n * obj.M) \ obj.Y;
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