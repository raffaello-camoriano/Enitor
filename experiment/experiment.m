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

            % Get training set (Xtr, Ytr) from dataset
            Xtr = ds.X(ds.trainIdx,:);
            Ytr = ds.Y(ds.trainIdx,:);
            Xte = ds.X(ds.testIdx,:);
            Yte = ds.Y(ds.testIdx,:);
            
            % get handle to the performanceMeasure function specific of the
            % dataset, to be passed to the algo.train method
            performanceMeasure = @ds.performanceMeasure;
            
            algo.train( Xtr , Ytr, performanceMeasure , true , 0.2);
            obj.result.Ypred = algo.test(Xte);
                                    
            obj.result.perf = ds.performanceMeasure( Yte , obj.result.Ypred );
        end
    end
    
end

