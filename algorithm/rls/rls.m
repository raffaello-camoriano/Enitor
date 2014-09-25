classdef rls < algorithm
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        kernelType
        kernel
        W % Weights matrix
    end
    
    methods 
        function init( obj , kerTy )
            obj.kernelType = kerTy;
        end
        
        function train(obj , X , Y)
            obj.W = ...;    %% TODO: implement
        end
        
        function Ypred = test( obj , X )

            Ypred = obj.W * X;
        end
        
        function crossVal(obj , X , Y)
            if ( strcmp(this.kernelType , 'gaussianKernel') )
                this.kernel = gaussianKernel.init();
            end
        end
    end
end

