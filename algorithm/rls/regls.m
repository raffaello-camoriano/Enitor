classdef regls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kernelType
        kernel
        kerParStar
        filterParStar
        c
        W % Weights matrix
        numFolds
        numKerParGuesses
    end
    
    methods
        
        function obj = regls(kerTy , numFolds , numKerParGuesses)
            obj.kernelType = kerTy;
            
            obj.numFolds = numFolds;
            if numFolds < 2
                display('Minimum number of folds: 2. numFolds set to 2.')
                obj.numFolds = 2;
            end

            obj.numKerParGuesses = numKerParGuesses;
        end
        
        function init( obj , kerTy )
            obj.kernelType = kerTy;
        end
        
        function train(obj , X , Y, sigma, lambda)

            % Get kernel type and instantiate kernel (including sigma)
            if strcmp(obj.kernelType , 'gaussian');
                obj.kernel = gaussianKernel(X , X , sigma);
            end
            
            R =chol(obj.kernel.K+lambda*eye(size(obj.kernel)));
            obj.c = (R\(R'\Y));
            
            %obj.W = inv( obj.kernel );    %% TODO: implement
        end
        
        function Ypred = test( obj , Xtr, Xte , sigma)

            % Get kernel type and instantiate kernel (including sigma)
            if strcmp(obj.kernelType , 'gaussian');
                trainTestKer = gaussianKernel(Xte , Xtr , sigma);
            end

            Ypred = trainTestKer.K * obj.c;
            
        end
        
        function crossVal(obj , dataset)
            
            if obj.numFolds > dataset.nTr
                display(['Maximum number of folds:' dataset.nTr '. numFolds set to nTr.'])
                obj.numFolds = dataset.nTr;
            end
            
%             if ( strcmp(obj.kernelType , 'gaussianKernel') )
%                 
%             else
%                 error('Kernel type not implemented.');
%             end

            % Performance measure storage variable
            perfMeas = zeros(obj.numFolds , obj.numKerParGuesses);
            
            % Get kernel type and instantiate kernel
            if strcmp(obj.kernelType , 'gaussian');
                obj.kernel = gaussianKernel( dataset.X , dataset.X );
            end
            
            % Get guesses vector for the sigma kernel parameter
            kerParGuesses = obj.kernel.range(obj.numKerParGuesses);
                
            for i = 1:obj.numFolds
                for j = 1:obj.numKerParGuesses
                
                    i
                    j
                    
                    sgm = kerParGuesses(j);
                    
                    testFoldIdx = round(dataset.nTr/obj.numFolds)*(i-1) + 1 : round(dataset.nTr/obj.numFolds)*i;                
                    trainFoldIdx = setdiff(dataset.trainIdx, testFoldIdx);

                    lam = 1;    % Lambda = 1, just for debug
                    
                    obj.train(dataset.X(trainFoldIdx,:) , dataset.Y(trainFoldIdx), sgm, lam);
                    Ypred = obj.test( dataset.X(trainFoldIdx,:) , dataset.X(testFoldIdx,:) , sgm);

                    perfMeas(i,j) = dataset.performanceMeasure(dataset.Y(testFoldIdx) , Ypred);

                end
            end
            
            % Set the index of the best kernel parameter
            [~,kerParStarIdx] = min(median(perfMeas,1));
            obj.kerParStar = kerParGuesses(kerParStarIdx);
            
            % Set the index of the best filter parameter
            obj.filterParStar = 1;  % DEBUG
            
            % Compute the kernel with the best kernel parameter
            obj.kernel.compute(obj.kerParStar);
            
        end
    end
end

