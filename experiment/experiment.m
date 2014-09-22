classdef experiment < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        name
    end
    
    methods
        
        function obj = experiment( name_)
            obj.name = name_;
        end
        
        function run (obj , algo , trainValSet , testSet)
            assert(isa(algo,'algorithm') , '1st argument is not of class algorithm');
            assert(isa(trainValSet,'dataset') , '2nd argument is not of class dataset');
            assert(isa(testSet,'dataset') , '3rd argument is not of class dataset');

            algo.train(trainValSet);
            algo.test(testSet);
        end
    end
    
end

