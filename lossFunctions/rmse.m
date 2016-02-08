function perf = rmse( Y , ~, Ypred )
%CLASSIFICATIONERROR 
%   Y: true labels
%   Ypred: predicted labels
            
    perf = sqrt(sum((Y - Ypred).^2)/size(Y,1));
end
