classdef kernel < handle
    %KERNEL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods ( Abstract )
        %obj = kernel();
        init(obj , X , Y);
        compute(obj , sigma);
    end
end

