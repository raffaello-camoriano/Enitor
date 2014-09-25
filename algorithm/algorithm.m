
classdef algorithm < handle

   methods (Abstract)
      init(obj);
      train(obj , X , Y);
      Ypred = test(obj , X );
      crossVal(obj , X , Y);
      
   end % methods
end % classdef