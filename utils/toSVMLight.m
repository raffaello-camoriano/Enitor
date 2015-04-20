% This script converts a binary classification dataset composed of a dense
% matrix X and a labels vector Y into a .dat file compatible with the
% SVMlight software

% X = ds.X(ds.trainIdx,:);
% Y = ds.Y(ds.trainIdx,:);

X = ds.X(ds.testIdx,:);
Y = ds.Y(ds.testIdx,:);
n = size(X,1);
d  =  size(X,2);

data = cell(n,d+1);

% Fill cell array
for i = 1:n
    
    % Write labels
    data{i,1} = num2str(Y(i));
   
    % Write features
    for j = 1:d
        data{i,j+1} = [ num2str(j) , ':' , num2str(X(i,j))];
    end
end

T = cell2table(data);
writetable(T , 'dataset_SVMlight.dat', 'WriteVariableNames' , false , 'Delimiter',' ');
    