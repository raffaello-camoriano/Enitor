
classdef InsuranceCompanyBenchmark < dataset
   
   properties
        outputFormat
   end
   
   methods
        function obj = InsuranceCompanyBenchmark(nTr , nTe, outputFormat, shuffleTraining, shuffleTest, shuffleAll)

            % Call superclass constructor with arguments
            obj = obj@dataset([], shuffleTraining, shuffleTest, shuffleAll);
            
			obj.d = 85;        % Fixed size for the full dataset
            dataTrain = load('ticdata2000.txt');
                   
            obj.X = dataTrain(:,1:obj.d);
            obj.Y = dataTrain(:,end);
                            
            obj.nTrTot = size(obj.X,1);
            
            dataTest = load('ticeval2000.txt');
            labelsTest = load('tictgts2000.txt');
            
            obj.X = [obj.X ; dataTest(:,1:obj.d)];
            obj.Y = [obj.Y ; labelsTest];
            
            obj.nTeTot = size(dataTest,1);
            obj.n = size(obj.Y , 1);

            obj.t = 2;
                
            if isempty(nTr) || isempty(nTe) 

                obj.nTr = obj.nTrTot;
                obj.nTe = obj.nTeTot;
                
                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr + 1 : obj.nTr + obj.nTe;
                
            else
                
                if (nTr < 2) || (nTe < 1) ||(nTr > obj.nTrTot) || (nTe > obj.nTeTot)
                    error('(nTr < 2) || (nTe < 1) ||(nTr > obj.nTrTot) || (nTe > obj.nTeTot)');
                end
                
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                tmp = randperm(obj.nTrTot);                            
                obj.trainIdx = tmp(1:obj.nTr);          
                
                tmp = obj.nTrTot + randperm( obj.nTeTot );
                obj.testIdx = tmp(1:obj.nTe);
            end
            
            obj.X = obj.scale(obj.X);
%             obj.X = 2*obj.X -1;
            
            % Shuffling

            obj.shuffleTraining = shuffleTraining;
            if shuffleTraining == 1
                obj.shuffleTrainIdx();
            end
            
            obj.shuffleTest = shuffleTest;
            if shuffleTest == 1
                obj.shuffleTestIdx();
            end
            
            obj.shuffleAll = shuffleAll;
            if shuffleAll == 1
                obj.shuffleAllIdx();
            end
            
            % Reformat output columns
            if (nargin > 2) && (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
                obj.outputFormat = outputFormat;
            else
                display('Wrong or missing output format, set to plusMinusOne by default');
                obj.outputFormat = 'plusMinusOne';
            end
            
            if strcmp(obj.outputFormat, 'plusMinusOne')
                obj.Y = obj.Y*2-1;
            elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
                obj.Y = obj.Y*2-1;
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
            
            Ypred(Yscores>0) = 1;
        end
            
        % Compute performance measure on the given outputs according to the
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            
%             Ypred = obj.scoresToClasses(Ypred);

            % RMSE
            perf = sqrt(sum((Y - Ypred).^2)/size(Y,1));

            % Classification accuracy
%             correctIdx = Y == Ypred;
%             numCorrect = sum(correctIdx);
%             
%             perf = 1 - (numCorrect / size(Y,1));
        end
        
        % Scales matrix M between -1 and 1
        function Ms = scale(obj , M)

            mx = max(M);
            mn = min(M);
            
            delta = mx - mn;
            Ms = transpose(bsxfun(@minus, M', mn'));
            Ms = transpose(bsxfun(@rdivide, Ms', delta'));
            Ms = Ms*2 - 1;
        end  
        
        function getTrainSet(obj)
            
        end
        
        function getTestSet(obj)
            
        end
        
   end % methods
end % classdef
