function perf = rmse( Y , Yscores, ~ )
%CLASSIFICATIONERROR 
%   Y: true labels
%   Ypred: predicted labels
            
    perf = sqrt(sum((Y - Yscores).^2)/size(Y,1));
end
