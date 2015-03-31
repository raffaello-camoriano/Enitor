
classdef Adult < dataset
   
   properties
        outputFormat
   end
   
   methods
        function obj = Adult(nTr , nTe, outputFormat)
            
%             trainData = readtable('adult.data.csv' , 'Delimiter',',','ReadVariableNames',false );
%             testData = readtable('adult.test.csv' , 'Delimiter',',','ReadVariableNames',false , 'HeaderLines' , 1);
%             fid = fopen('adult.attributenames.csv');
%             attrNames = textscan(fid, '%s');
%             fclose(fid);
            

            display('This implementation of the Adult dataset considers all the available attributes, (d = 123)');
            obj.d = 123;        % Fixed size for the full dataset
            data = load('adult.mat');
            
            % Fix dimensionality issues
            tesz = size(data.testing_vectors,2);
            if tesz < obj.d
                data.testing_vectors = [data.testing_vectors , zeros(size(data.testing_vectors,1), obj.d - tesz)];
            end
            
            trsz = size(data.training_vectors,2);
            if trsz < obj.d
                data.testing_vectors = [data.testing_vectors , zeros(size(data.testing_vectors,1), obj.d - trsz)];
            end            
            
            obj.X = [data.training_vectors ; data.testing_vectors];
            obj.X = obj.scale(obj.X);
            obj.Y = [data.training_labels ; data.testing_labels];
                            
            obj.nTrTot = size(data.training_labels,1);
            obj.nTeTot = size(data.testing_labels,1);
            obj.n = size(obj.Y , 1);

            obj.t = 2;
                
            if nargin == 0

                obj.nTr = obj.nTrTot;
                obj.nTe = obj.nTeTot;
                
                obj.trainIdx = 1:obj.nTr;
                obj.testIdx = obj.nTr + 1 : obj.nTr + obj.nTe;
                
            elseif (nargin >1)
                
                if (nTr < 2) || (nTe < 1) ||(nTr > obj.nTrTot) || (nTe > obj.nTeTot)
                    error('(nTr < 2) || (nTe < 1) ||(nTr > obj.nTrTot) || (nTe > obj.nTeTot)');
                end
                
                obj.nTr = nTr;
                obj.nTe = nTe;
                
                tmp = randperm( obj.nTrTot);                            
                obj.trainIdx = tmp(1:obj.nTr);          
                
                tmp = obj.nTrTot + randperm( obj.nTeTot );
                obj.testIdx = tmp(1:obj.nTe);
            end
            
            obj.shuffleTrainIdx();
            obj.shuffleTestIdx();
            
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
            
%             for i = 1:size(Ypred,1)
%                 if Yscores(i) > 0
%                     Ypred(i) = 1;
%                 end
%             end
            
            Ypred(Yscores>0) = 1;
        end
            
        % Compute performance measure on the given outputs according to the
        % USPS dataset-specific ranking standard measure
        function perf = performanceMeasure(obj , Y , Ypred , varargin)
            
            % Check if Ypred is real-valued. If yes, convert it.
%             if obj.hasRealValues(Ypred)
                Ypred = obj.scoresToClasses(Ypred);
%             end
            
            % Compute error rate
%             numCorrect = 0;
            
%             for i = 1:size(Y,1)
%                 if Y(i) == Ypred(i)
%                     numCorrect = numCorrect +1;
%                 end
%             end
            
            correctIdx = Y == Ypred;
            numCorrect = sum(correctIdx);
            
            perf = 1 - (numCorrect / size(Y,1));
        end
        
        % Scales matrix M between -1 and 1
        function Ms = scale(obj , M)
            
            mx = max(max(M));
            mn = min(min(M));
            
            Ms = ((M + abs(mn)) / (mx - mn)) * 2 - 1;
            
        end
        
        function getTrainSet(obj)
            
        end
        
        function getTestSet(obj)
            
        end
        
   end % methods
end % classdef