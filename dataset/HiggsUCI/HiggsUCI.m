
classdef HiggsUCI < dataset
   
   properties
        %outputFormat
        
        weights
        EventId
        outputFormat
        
   end
   
   methods
        function obj = HiggsUCI(nTr , nTe, outputFormat, shuffleTraining, shuffleTest, shuffleAll)
                        
            % Call superclass constructor with arguments
            obj = obj@dataset([], shuffleTraining, shuffleTest, shuffleAll);
            
            % Store samples
            MF = matfile('Higgs.mat');
            
            if isempty(nTr)
                
                obj.nTr = 2^23;
                display('Default number of training points = 2^23');
                
                obj.nTe = 2^21;
                display('Default number of test points = 2^21');
                
            else
                
                obj.nTr = 2^floor(log2(nTr));
                if log2(nTr) > 23
                   error('Max number of training points = 2^23');
                end
                display([ 'Number of training points = ' , num2str(obj.nTr) ]);
                
                obj.nTe = 2^floor(log2(nTe));
                if log2(nTe) > 21
                   error('Max number of test points = 2^21');
                end
                display([ 'Number of test points = ' , num2str(obj.nTe) ]);
            end
            
            obj.n = obj.nTr + obj.nTe;
                        
            obj.X = MF.X(1:obj.n,2:end);
            obj.Y = MF.X(1:obj.n,1);
            
            % If necessary, balance obj.Y
%             if (abs(sum(obj.Y))/length(obj.Y)) > 0.5 % NOTE: arbitrary threshold
%                 obj.Y = obj.balanceLabels(obj.Y);
%             end
            
            obj.d = size(obj.X , 2);
            obj.t = size(obj.Y , 2);
                
                
            obj.trainIdx = 1:obj.nTr;
            obj.testIdx = obj.nTr + 1 : obj.n;
                        
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
            obj.problemType = 'classification';
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
        
        function Ybal = balanceLabels(obj, Y)
            
            nplus = sum(Y == 1);
            nminus = sum(Y == -1);
        
            Ybal = Y;
            
            if nplus > nminus
               Ybal(Y == -1) =  (nplus/nminus) * Y(Y == -1);
%                Ybal(Y == 1) =  1;
            else
               Ybal(Y == 1) =  (nminus/nplus) * Y(Y == 1);
%                Ybal(Y == -1) =  -1;
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

%             Ypred = sign(Yscores);

        end
            
        % Compute performance measure on the given outputs according to the
        % dataset-specific ranking standard measure
        %function perf = performanceMeasure(obj , Y , Ypred)
        function perf = performanceMeasure(obj, Y , Ypred, Yidx)
            
            % Yidx is the vector of row indexes on the dataset Y matrix to
            % which prediction matrix Ypred corresponds
            
            if (length(Yidx) ~= size(Ypred,1)) || (length(Yidx) ~= size(Y,1))
                error('Ground truth cardinality and predictions cardinality do not match.');
            end
               
            % Check if Ypred is real-valued. If yes, convert it.
%             if obj.hasRealValues(Ypred)
                %Yscores = Ypred;
                Ypred = obj.scoresToClasses(Ypred);
%             end
                        
%             % Check if Y is real-valued. If yes, convert it.
%             if obj.hasRealValues(Y)
%                 Y = obj.scoresToClasses(Y);
%             end
            
            [~,~,~,perf] = perfcurve(Y,Ypred,+1);  

        end
    end % methods
end % classdef