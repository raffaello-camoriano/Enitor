function perf = classificationError( Y , ~, Ypred )
%CLASSIFICATIONERROR 
%   Y: true labels
%   Ypred: predicted labels
            
    perf = 1 - (sum(Y == Ypred) / size(Y,1));
end
