classdef experiment < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        name
        result
    end
    
    methods
        
        function obj = experiment( name_)
            obj.name = name_;
        end
        
        function obj = run (obj , algo , ds)
            assert(isa(algo,'algorithm') , '1st argument is not of class algorithm');
            assert(isa(ds,'dataset') , '2nd argument is not of class dataset');

            algo.crossVal(ds);
            
            algo.train(ds.X(ds.trainIdx) , ds.Y(ds.trainIdx), algo.kerParStar, algo.filterParStar);
            obj.result.Ypred = algo.test(ds.X(ds.trainIdx,:) , ds.X(ds.testIdx,:) , algo.kerParStar);
                        
            obj.result.perf = ds.performanceMeasure( ds.Y(ds.testIdx) , obj.result.Ypred );
        end
    end
    
end

