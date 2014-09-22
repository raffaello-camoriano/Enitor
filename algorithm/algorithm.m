
classdef algorithm < handle
   
   % Define an event
   events
       
   end
   
   methods (Abstract)
      init(obj);
      train(obj , X , Y);
      test(obj , X , Y );
      crossVal(obj , X , Y);
      
   end % methods
end % classdef