
classdef pumadyn < dataset
   
   properties
        outputFormat
        linearity
        noise
   end
   
   methods
        %     d : An integlue signifying the number of input attributes in each case, for example `32'.
        %     linearity: One of the characters `f' or `n' signifying `fairly linear' or `non-linear', respectively.
        %     noise: One of the charer vaacters `m' or `h' signifying `medium unpredictability/noise' or `high unpredictability/noise', respectively.
        function obj = pumadyn(nTr , nTe, d , linearity , noise)
            
            if  isvarname('d') && (d == 8 || d == 32)
                obj.d = d;
            else
                warning('"d" should be set to 8 or 32. Set to 8 by default.');
                obj.d = 8;
            end
            if isvarname('linearity') && (strcmp(linearity, 'f') || strcmp(linearity, 'n'))
                obj.linearity = linearity;
            else
                warning('"linearity" should be set to "f" (fairly linear) or "n" (non-linear). Set to "f" by default.');
                obj.linearity = 'f';
            end
            if isvarname('noise') && (strcmp(noise, 'm') || strcmp(noise, 'h'))
                obj.noise = noise;
            else
                warning('"noise" should be set to "m" (medium) or "h" (high). Set to "m" by default.');
                obj.noise = 'm';
            end
            
            % Assemble file name string
            fName = strcat('pumadyn-' , num2str(obj.d) , obj.linearity , obj.noise , '/Dataset.data' );
                
            data = load(fName);
            
            obj.X = data(:,1:obj.d);
            obj.Y = data(:,obj.d+1);

            obj.X = obj.scale(obj.X);   
            
            obj.n = size(obj.X , 1);
            obj.nTrTot = 4096;
            obj.nTeTot = 4096;
            
            if nargin < 2

                warning('Specify number of training and testing samples. nTr = 2000; nTe = 4096 specified by default.');
                
                obj.nTr = 2000;
                obj.nTe = 4096;
                
                % Select consecutive samples
                obj.trainIdx = 1:nTr;          
                obj.testIdx = obj.nTrTot + 1 : obj.nTrTot + nTe;      
            
            elseif (nargin > 1)
                
                if (nTr < 2) || (nTe < 1) || (nTr > obj.nTrTot) || (nTe > obj.nTeTot)
                    error('(nTr < 2) || (nTe < 1) || (nTr > obj.nTrTot) || (nTe > obj.nTeTot)');
                end
                
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                % Select consecutive samples
                obj.trainIdx = 1:nTr;          
                obj.testIdx = obj.nTrTot + 1 : obj.nTrTot + nTe;          
                
                %tmp = randperm( obj.nTrTot);                            
                %obj.trainIdx = tmp(1:obj.nTr);          
                
                %tmp = obj.nTrTot + randperm( obj.nTeTot );
                %obj.testIdx = tmp(1:obj.nTe);
            end
            
%             obj.shuffleTrainIdx();
%             obj.shuffleTestIdx();
            
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
%         function Ypred = scoresToClasses(obj , Yscores)    
%             
%             if strcmp(obj.outputFormat, 'zeroOne')
%                 Ypred = zeros(size(Yscores));
%             elseif strcmp(obj.outputFormat, 'plusMinusOne')
%                 Ypred = -1 * ones(size(Yscores));
%             elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
%                 Ypred = -1/(obj.t - 1) * ones(size(Yscores));
%             end
%             
%             for i = 1:size(Ypred,1)
%                 [~,maxIdx] = max(Yscores(i,:));
%                 Ypred(i,maxIdx) = 1;
%             end
%         end
            
        % Compute performance measure on the given outputs
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            % RMSE
            perf = sqrt(sum((Y - Ypred).^2)/size(Y,1));
        end
        
        % Scales matrix M between -1 and 1
        function Ms = scale(obj , M)
            
%             minVal = double([-95 0 -37 15 -50 -50 -50 -50 -200 -200 -200 -200]);
%             maxVal = double([10 161 80 106 50 50 50 50 200 200 200 200]);
            minVal = min(M,[],1);
            maxVal = max(M,[],1);
            
            Ms = zeros(size(M));
            for i = 1:size(M,2)
             
                Ms(:,i) = ((abs(minVal(i)) + M(:,i)  ) / (maxVal(i) - minVal(i))) * 2 - 1;
                
            end
        end        
   end % methods
end % classdef