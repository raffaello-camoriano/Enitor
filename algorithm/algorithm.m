
classdef algorithm < handle

    properties
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