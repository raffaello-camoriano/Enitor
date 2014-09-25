classdef tikhonov < filter
    %TIKHONOV Summary of this class goes here
    %   Detailed explanation goes here

    properties
        U, T, Y0
        rng, Yreg
    end
    
    methods ( Abstract )
        %obj = kernel();
        function init(obj , K , Y)
            [obj.U, obj.T] = hess(K);
            obj.Y0 = obj.U'*Y;
        end
        
        function range(obj, numGuesses)
            
            % TODO: @Ale: Smart computation of eigmin and eigmax starting from the
            % tridiagonal matrix U
            
            % ...
            
            % GURLS code below, set 'eigmax' and 'eigmin' variables
            
            % maximum eigenvalue
            lmax = eigmax;

            % just in case, when r = min(n,d) and r x r has some zero eigenvalues
            % we take a max; 200*sqrt(eps) is the legacy number used in the previous
            % code, so i am just continuing it.

            lmin = max(min(lmax*opt.smallnumber, eigmin), 200*sqrt(eps));

            powers = linspace(0,1,opt.nlambda);
            obj.rng = lmin.*(lmax/lmin).^(powers);
            obj.rng = obj.rng/n;
            
        end
        
        function compute(obj , lambda)
            n = size(obj.T,1);
            obj.Yreg = obj.U*((obj.T+lambda*n*eye(n))\obj.Y0);
        end
    end
end