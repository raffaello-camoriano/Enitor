classdef gaussianKernel < kernel
    %GAUSSIAN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        SqDistMat
        n
        m
    end
    
    methods
        
        function obj = gaussianKernel( X1 , X2 , sigma )
            
            obj.n = size(X1 , 1);
            obj.m = size(X2 , 1);
            obj.computeSqDistMat(X1,X2);
            
            if  nargin > 2
                obj.compute(sigma);
            end
        end
        
        % Computes the squared distance matrix SqDistMat based on X1, X2
        function computeSqDistMat(obj , X1 , X2)
            
            Sx1 = sum( X1.*X1 , 2);
            Sx2 = sum( X2.*X2 , 2)';
            Sx1x2 = X1 * X2';
            
            obj.SqDistMat = repmat(Sx1 , 1 , obj.m) -2*Sx1x2 + repmat(Sx2 , obj.n , 1);
        
        end
        
        % Computes the kernel matrix SqDistMat based on SqDistMat and sigma
        function compute(obj , sigma)
            if( nargin > 1 )
                obj.K = exp(-obj.SqDistMat/(2*sigma^2));
            else
                disp('sigma parameter not specified! Exiting...');
            end
        end
        
        % Computes the range for the sigma parameter guesses
        function rng = range(obj , numGuesses)
            if( nargin < 2 )
                numGuesses = 20;    % Default guesses number
                disp('Number of guesses not specified! Set to 20 by default...');
            end
            
            if (obj.m ~= obj.n)
                error('Error, the distance matrix is not squared! Aborting...');
            end
            
            % Compute max and min sigma guesses (same strategy used by
            % GURLS)
            
            D = sort(obj.SqDistMat(tril(true(obj.n),-1)));
            firstPercentile = round(0.01*numel(D)+0.5);
            minGuess = sqrt(D(firstPercentile));

            %D = sort(opt.kernel.distance);
            %opt.sigmamax = median(D(n,:));
            maxGuess = sqrt(max(max(obj.SqDistMat)));

            if minGuess <= 0
                minGuess = eps;
            end
            if maxGuess <= 0
                maxGuess = eps;
            end	
            
            rng = linspace(minGuess, maxGuess , numGuesses);
        end        
    end
end

