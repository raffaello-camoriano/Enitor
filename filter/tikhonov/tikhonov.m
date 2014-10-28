classdef tikhonov < filter
    %TIKHONOV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        U, T, Y0
        coeffs
        n % Number of samples
    end
    
    methods
        
        function obj = tikhonov( K , Y )
            obj.init(K , Y );
        end
        
        function init(obj , K , Y )
            [obj.U, obj.T] = hess(K);
            
            obj.Y0 = obj.U'*Y;
            
            obj.n = size(K,1);
        end
        
        function rng = range(obj, numGuesses)
            
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

            powers = linspace(0,1,numGuesses);
            rng = (lmin.*(lmax/lmin).^(powers))/obj.n;            
        end
        
        function compute(obj , lambda)

            obj.coeffs = obj.U*((obj.T+lambda*obj.n*eye(obj.n))\obj.Y0);
        end
    end
end