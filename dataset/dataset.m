
classdef dataset < handle
   
   properties
       
        problemType     % regression or classification
        
        n
        nTr
        nTe
        
        nTot
        nTrTot
        nTeTot
        d
        t

        X
        Y

        trainIdx
        testIdx
        
        shuffleTraining
        shuffleTest
        shuffleAll

        lossFunction
   end
   
   methods
        
        function obj = dataset(fname, shuffleTraining, shuffleTest, shuffleAll)
            if  ~isempty(fname)            
                data = load(fname);
                obj.X = data.X;
                obj.Y = data.Y;
                obj.n = size(data.X , 1);
                obj.d = size(data.X , 2);
                obj.t = size(data.y , 2);
                
                % Set problem type

                obj.problemType = 'classification';
                for i = 1:size(obj.Y,1)
                    for j = 1:size(obj.Y,2)

                        if mod(obj.Y(i,j),1) ~= 0
                            obj.problemType = 'regression';
                        end
                    end
                end
            end
            
            % Shuffling

            if  ~isempty(shuffleTraining)            
                obj.shuffleTraining = shuffleTraining;
            else
                obj.shuffleTraining = 0;
            end
            
            if  ~isempty(shuffleTest)
                obj.shuffleTest = shuffleTest;
            else
                obj.shuffleTest = 0;
            end
            
            if  ~isempty(shuffleAll)
                obj.shuffleAll = shuffleAll;
            else
                obj.shuffleAll = 0;
            end
            
%             if shuffleTraining == 1
%                 obj.shuffleTrainIdx();
%             end
%             if shuffleTest == 1
%                 obj.shuffleTestIdx();
%             end
%             if shuffleAll == 1
%                 obj.shuffleAllIdx();
%             end
        end

        
        % Compute random permutation of the training set indexes
        function shuffleTrainIdx(obj)
%             obj.trainIdx = obj.trainIdx(randperm(obj.nTr));

            tmp = randperm(obj.nTrTot);
            obj.trainIdx = tmp(1:obj.nTr);
        
        end
        
        % Compute random permutation of the test set indexes
        function shuffleTestIdx(obj)
%             obj.testIdx = obj.testIdx(randperm(obj.nTe));
            
            tmp = randperm(obj.nTeTot);
            obj.testIdx = obj.nTrTot + tmp(1:obj.nTe);
        end
        
        % Compute random permutation of all indexes
        function shuffleAllIdx(obj)
            tmp = [obj.trainIdx obj.testIdx];
            tmp2 = tmp(randperm(obj.nTr+obj.nTe));
            obj.trainIdx = tmp2(1:obj.nTr);
            obj.testIdx = tmp2(obj.nTr+1:obj.nTr+obj.nTe);
        end
    end % methods
end % classdef
