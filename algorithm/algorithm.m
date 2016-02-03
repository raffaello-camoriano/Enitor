
classdef algorithm < handle

    properties
        
        % I/O options
        storeFullTrainPerf  % Store full training performance matrix 1/0
        storeFullTrainPred  % Store full training predictions matrix 1/0
        storeFullValPerf    % Store full validation performance matrix 1/0
        storeFullValPred    % Store full validation performance matrix 1/0
        storeFullTestPerf   % Store full test performance matrix 1/0
        storeFullTestPred   % Store full test predictions matrix 1/0
        
        valPerformance      % Validation performance matrix
        valPred             % Validation predictions matrix
        trainPerformance    % Training performance matrix
        trainPred           % Training predictions matrix
        testPerformance     % Test performance matrix
        testPred            % Test predictions matrix
        
        verbose 
    end
    
    methods (Abstract)
      init(obj);
      train(obj , X , Y);
      Ypred = test(obj , X );
      %crossVal(obj , X , Y);   % cross validation is included in the train
      %method
    end % methods
end % classdef