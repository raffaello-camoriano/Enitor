
classdef ReutersMoneyFX < dataset
   
   properties
        outputFormat
   end
   
   methods
        function obj = ReutersMoneyFX(nTr , nTe, outputFormat)
            
%             % Load training set and format it
%             fileID = fopen('money-fx.trn');
%             formatSpec = '%s';
%             data = textscan(fileID,formatSpec, ...            
%                             'Delimiter', ' ', ...
%                             'CollectOutput', true);
%             fclose(fileID);
%             
%             obj.X = sparse(0,0);
%             obj.Y = [];
%             j = 0;  % Row counter
%             for i = 1:size(data{1,1},1)
%                 if strcmp(data{1,1}(i) , '+1') || strcmp(data{1,1}(i) , '-1')
%                     j = j+1;
%                     obj.Y(j,1) = str2double(data{1,1}(i));
%                 else
%                     tmp = cell2mat(data{1,1}(i));
%                     C = textscan(tmp,'%n:%f', 1);
%                     obj.X(j,C{1}) = C{2};
%                 end
%             end
%             
%             % Set training dimensionalities
%             obj.nTrTot = size(obj.X,1);
%             
% 
%             
%             % Load test set and format it
%             fileID = fopen('money-fx.tst');
%             formatSpec = '%s';
%             data = textscan(fileID,formatSpec, ...            
%                             'Delimiter', ' ', ...
%                             'CollectOutput', true);
%             fclose(fileID);
%             
%             for i = 1:size(data{1,1},1)
%                 if strcmp(data{1,1}(i) , '+1') || strcmp(data{1,1}(i) , '-1')
%                     j = j+1;
%                     obj.Y(j,1) = str2double(data{1,1}(i));
%                 else
%                     tmp = cell2mat(data{1,1}(i));
%                     C = textscan(tmp,'%n:%f', 1);
%                     obj.X(j,C{1}) = C{2};
%                 end
%             end            
%             
%             % Set test set dimensionalities
%             obj.n = size(obj.X , 1);
%             obj.d = size(obj.X , 2);
%             obj.t = 2;
%             obj.nTeTot = obj.n - obj.nTrTot;
            
            load('ReutersMoneyFXMatlab.mat');
            obj.X = X;
            obj.Y = Y;
            obj.d = d;
            obj.t = t;
            obj.n = n;
            obj.nTrTot = nTrTot;
            obj.nTeTot = nTeTot;
            clear X Y d t n nTrTot nTeTot

            % Shuffle training indexes
            shuffIdx = randperm(obj.nTrTot);
            obj.X(1:obj.nTrTot , :) = obj.X(shuffIdx,:);
            obj.Y(1:obj.nTrTot) = obj.Y(shuffIdx);
            
            % Shuffle test indexes
            shuffIdx = obj.nTrTot + randperm(obj.nTeTot);
            obj.X(obj.nTrTot+1 : obj.n , :) = obj.X(shuffIdx,:);
            obj.Y(obj.nTrTot+1 : obj.n) = obj.Y(shuffIdx);
            
            if nargin == 0
                
                obj.nTr = obj.nTrTot;
                obj.nTe = obj.nTeTot;

                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr+1:obj.nTr+obj.nTe;
                
            elseif (nargin >1) && (nTr > 1) && (nTe > 0)
            
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                if (nTr > obj.nTrTot) || (nTe > obj.nTeTot)
                    error('(nTr > obj.nTrTot) || (nTe > obj.nTeTot)');
                end
                
                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTrTot+1:obj.nTrTot+obj.nTe;
                
            end
            
            % Reformat output columns
            if (nargin > 2) && (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
                obj.outputFormat = outputFormat;
            else
                display('Wrong or missing output format, set to plusMinusOne by default');
                obj.outputFormat = 'plusMinusOne';
            end
            
            if strcmp(obj.outputFormat, 'zeroOne')
                obj.Y = obj.Y/2+0.5;
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
                if Yscores(i) > 0
                    Ypred(i) = 1;
                end
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