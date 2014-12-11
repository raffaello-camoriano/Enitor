
classdef USPS_binary < dataset
   
   properties
        outputFormat
   end
   
   methods
        function obj = USPS_binary(nTr , nTe, outputFormat)
            
            data = load('usps_all.mat');
            
            obj.X = data.data(:,:,1:2);
            obj.X = double([obj.X(:,:,1) , obj.X(:,:,2)]');

            % downsample images
            for i=1:size(obj.X,1)
                
                tmp1 = vec2mat(obj.X(i,:), sqrt(size(obj.X,2)));
                tmp2 = imresize(tmp1,1/2);
                Xtmp(i,:) = reshape(tmp2, [1,size(obj.X,2)/4]);
            end
            
            obj.X = Xtmp;
            
            % Normalize
            obj.X = obj.X / 127.5 - 1;
                        
            gnd = zeros(size(obj.X,1),1);
            gnd(1:1100) = 1;
            gnd(1101:2200) = 2;
            
            obj.n = size(obj.X , 1);
            obj.d = size(obj.X , 2);
            obj.t = max(gnd);
            
            % Shuffle X and gnd
            shuffIdx = randperm(obj.n);
            
            obj.X = obj.X(shuffIdx,:);
            gnd = gnd(shuffIdx);
            
            if nargin == 0
                
                obj.nTr = 2000;
                obj.nTe = 200;

                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr+1:obj.nTr+obj.nTe;
                
            elseif (nargin >1) && (nTr > 1) && (nTe > 0)
            
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                if (nTr > 2000) || (nTe > 200)
                    error('(nTr > 2000) || (nTe > 200)');
                end
                
                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = 2001:2000+obj.nTe;
                
            end
            
            % Reformat output columns
            if (nargin > 2) && (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
                obj.outputFormat = outputFormat;
            else
                display('Wrong or missing output format, set to plusMinusOne by default');
                obj.outputFormat = 'plusMinusOne';
            end
            
            if strcmp(obj.outputFormat, 'zeroOne')
                obj.Y = zeros(obj.n,obj.t);
            elseif strcmp(obj.outputFormat, 'plusMinusOne')
                obj.Y = -1 * ones(obj.n,obj.t);
            elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                obj.Y = -1/(obj.t - 1) * ones(obj.n,obj.t);
            end
               
            for i = 1:obj.n
                obj.Y(i , gnd(i)) = 1;
                % Display associated image
                %imshow(vec2mat(obj.X(i,:),16))
                %data.gnd(i)
            end
            
            % Set problem type
            if obj.hasRealValues(obj.Y)
                obj.problemType = 'regression';
            else
                obj.problemType = 'classification';
            end
        end
        
        % Checks if matrix Y contains real values. Useful for
        % discriminating between classification and regression, or between
        % predicted scores and classes
        function res = hasRealValues(obj , M)
        
            res = false;
            for i = 1:size(M,1)
                for j = 1:size(M,2)
                    if mod(M(i,j),1) ~= 0
                        res = true;
                    end
                end
            end
        end
        
        % Compute predictions matrix from real-valued scores matrix
        function Ypred = scoresToClasses(obj , Yscores)    
            
            if strcmp(obj.outputFormat, 'zeroOne')
                Ypred = zeros(size(Yscores));
            elseif strcmp(obj.outputFormat, 'plusMinusOne')
                Ypred = -1 * ones(size(Yscores));
            elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                Ypred = -1/(obj.t - 1) * ones(size(Yscores));
            end
            
            for i = 1:size(Ypred,1)
                [~,maxIdx] = max(Yscores(i,:));
                Ypred(i,maxIdx) = 1;
            end
        end
            
        % Compute performance measure on the given outputs according to the
        % USPS dataset-specific ranking standard measure
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            
            % Check if Ypred is real-valued. If yes, convert it.
            if obj.hasRealValues(Ypred)
                Ypred = obj.scoresToClasses(Ypred);
            end
            
            % Compute error rate
            numCorrect = 0;
            
            for i = 1:size(Y,1)
                if sum(Y(i,:) == Ypred(i,:)) == size(Y,2)
                    numCorrect = numCorrect + 1;
                end
            end
            
            perf = 1 - (numCorrect / size(Y,1));
        end
        
   end % methods
end % classdef