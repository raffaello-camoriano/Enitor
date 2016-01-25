function perf = hingeLoss( Y , Yscores, ~ )
%CLASSIFICATIONERROR 
%   Y: true labels
%   Yscores: predicted scores
            
    prod = Yscores .* Y;
    I = prod <= 1;
    perf = sum( - prod(I) ) / size(Y,1);
end
