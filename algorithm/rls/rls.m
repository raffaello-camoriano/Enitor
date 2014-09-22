classdef rlsTikhonov < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kernelType
        kernel
    end
    
    methods 
        function init( obj , kerTy )
            obj.kernelType = kerTy;
        end
        
        function train(obj , X , Y);
            
        end
        
        function test(obj , X , Y );

        end
        
        function crossVal(obj , X , Y)
            if ( strcmp(this.kernelType , 'gaussianKernel') )
                this.kernel = gaussianKernel.init();
            end
        end
    end
end

