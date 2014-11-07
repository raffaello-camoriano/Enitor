
classdef Higgs < dataset
   
   properties
        %outputFormat
   end
   
   methods
        function obj = Higgs(nTr , nTe)
            
            warning('Test labels not available!');
            
            trainTab = readtable('training.csv');
            testTab = readtable('training.csv');
                        
            Xtr = trainTab{:,2:end-2};
            Xte = testTab{:,2:31};
            
            tmp = trainTab{:,end};
            
            Ytr = zeros(size(tmp));
            Ytr(strcmp(tmp,'b')) = -1;
            Ytr(strcmp(tmp,'s')) = 1;
            
            obj.Y = Ytr;    % NO Yte available!
            
            % If necessary, balance obj.Y
%             if (abs(sum(obj.Y))/length(obj.Y)) > 0.5 % NOTE: arbitrary threshold
%                 obj.Y = obj.balanceLabels(obj.Y);
%             end
            
            obj.X = [Xtr ; Xte];

            %obj.n = size(Xtr , 1) + size(Xte , 1);
            obj.n = size(Xtr , 1);
            
            obj.d = size(Xtr , 2);
            obj.t = size(Ytr , 2);
                
            if nargin == 0
                
%                 obj.nTr = size(Xtr , 1);
%                 obj.nTe = size(Xte , 1);

                obj.nTr = 10000;
                obj.nTe = 2678;
                
                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr + 1 : obj.nTr + obj.nTe;
                
            elseif (nargin > 1) && (nTr > 1) && (nTe > 0)
                
                if (nTr > 10000) || (nTe > 2678)
                    error('(nTr > 10000) || (nTe > 2678)');
                end
                
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = 10001:10000 + obj.nTe;
            end
            
            % Reformat output columns
%             if (nargin > 2) && (strcmp(outputFormat, 'zeroOne') ||strcmp(outputFormat, 'plusMinusOne') ||strcmp(outputFormat, 'plusOneMinusBalanced'))
%                 obj.outputFormat = outputFormat;
%             else
%                 display('Wrong or missing output format, set to plusMinusOne by default');
%                 obj.outputFormat = 'plusMinusOne';
%             end
            
%             if strcmp(obj.outputFormat, 'zeroOne')
%                 obj.Y = zeros(obj.n,obj.t);
%             elseif strcmp(obj.outputFormat, 'plusMinusOne')
%                 obj.Y = -1 * ones(obj.n,obj.t);
%             elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
%                 obj.Y = -1/(obj.t - 1) * ones(obj.n,obj.t);
%             end
%                
%             for i = 1:obj.n
%                 obj.Y(i , data.gnd(i)) = 1;
%             end
            
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
            
%             if strcmp(obj.outputFormat, 'zeroOne')
%                 Ypred = zeros(size(Yscores));
%             elseif strcmp(obj.outputFormat, 'plusMinusOne')
%                 Ypred = -1 * ones(size(Yscores));
%             elseif strcmp(obj.outputFormat, 'plusOneMinusBalanced')
%                 Ypred = -1/(obj.t - 1) * ones(size(Yscores));
%             end

            Ypred = sign(Yscores);

        end
            
        % Compute performance measure on the given outputs according to the
        % dataset-specific ranking standard measure
        function perf = performanceMeasure(obj , Y , Ypred)
            
            % Check if Ypred is real-valued. If yes, convert it.
            if obj.hasRealValues(Ypred)
                %Yscores = Ypred;
                Ypred = obj.scoresToClasses(Ypred);
            end
            % Check if Y is real-valued. If yes, convert it.
            if obj.hasRealValues(Ypred)
                Y = obj.scoresToClasses(Y);
            end
            
%             %Create submission table and save it to file
%             submissionTable = table([],[],Ypred, ...
%                          'VariableNames',{'EventId' 'RankOrder' 'Class'})
%             
%             writetable(submissionTable, 'dataset/Higgs/tmp/submissionFile.csv');
%             
%             %Create solution table and save it to file
%             solutionTable = table([],Y, [], ...
%                          'VariableNames',{ 'EventId', 'Class', 'Weight'})
%             
%             writetable(solutionTable , 'dataset/Higgs/tmp/solutionFile.csv');
%             
%             % Compute AMS
%             [status, perf]  = system('dataset/Higgs/HiggsBosonCompetition_AMSMetric_rev1.py dataset/Higgs/tmp/solutionFile.csv dataset/Higgs/tmp/submissionFile.csv');
%             
%             if status ~= 0
%                 error('Python AMS computation failed');
%             end  
            
            % Compute error rate
            numCorrect = 0;
            
            for i = 1:size(Y,1)
                if Y(i) == Ypred(i)
                    numCorrect = numCorrect +1;
                end
            end
            
            perf = 1 - (numCorrect / size(Y,1));            
        end
    end % methods
end % classdef