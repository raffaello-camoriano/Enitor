classdef randomFeaturesMapper
    %RANDOMFEATURES Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        mappingType
        d
        numRF
        proj
    end
    
    methods
        function obj = randomFeaturesMapper( d , numRF , mappingType)
            obj.init(d, unmRF, mappingType);
        end
        
        function obj = init(obj,d,numRF,mappingType)
            
            if strcmp(mappingType,'gaussian')
                obj.mappingType = mappingType;
            else
                error('mapping Type not available');
            end
            
            obj.numRF = numRF;
            obj.d = d;
            obj.proj = generateProj();            
        end
        
        function obj = generateProj(obj)
            
            if strcmp(obj.mappingType,'gaussian')
                obj.proj = sqrt(2) * randn(obj.d, obj.numRF);
            else
                error('mapping type not available');
            end
        end
        
        function mappedSample = map(inputSample)
            V = inputSample * obj.proj;
            mappedSample = [cos(V); sin(V)];
        end
    end
end

