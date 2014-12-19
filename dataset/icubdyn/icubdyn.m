
classdef icubdyn < dataset
   
   properties
        outputFormat
   end
   
   methods
        function obj = icubdyn(nTr , nTe)
            
            data = load('icubdyn.dat');
            
            obj.X = data(:,1:12);
%            obj.Y = data(:,13:18);
            obj.Y = data(:,13:15);

%             obj.nTr = size(obj.X,1);
%             obj.nTrTot = obj.nTr;
            
            obj.X = obj.scale(obj.X);   
            
%             obj.nTe = obj.n - obj.nTr;
%             obj.nTeTot = obj.nTe;
            
            obj.n = size(obj.X , 1);
            obj.d = size(obj.X , 2);
%             obj.t = 10;
                
            if nargin < 2

                error('Specify number of training and testing samples');
                
            elseif (nargin > 1)
                
                if (nTr < 2) || (nTe < 1) ||(nTr + nTe > obj.n)
                    error('(nTr < 2) || (nTe < 1) ||(nTr + nTe > obj.n)');
                end
                
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                % Select consecutive samples
                obj.trainIdx = 1:nTr;          
                obj.testIdx = nTr + 1 : nTr + nTe;          
                
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

            warning('The performance is currently averaged across the outputs.');
            perf = mean(sqrt(sum((Y - Ypred).^2)/size(Y,1)));
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
        
        function getTrainSet(obj)
            
        end
        
        function getTestSet(obj)
            
        end
        
   end % methods
end % classdef