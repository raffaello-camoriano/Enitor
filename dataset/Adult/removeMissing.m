% This script removes the rows of the specified full Adult dataset
% containing missing values

clear all;

fname = 'adult.test2.csv';
fnameClean = 'adult.test.clean.csv';

t = readtable(fname , 'ReadVariableNames',false , 'Format' , '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s');

a = t{:,:};

dirtyRows = [];     % Indexes of the rows to be removed
for i = 1:size(a,1)
    for j = 1:size(a,2)
        if strcmp(a(i,j) , '?')  == 1
            dirtyRows  = [dirtyRows , i];
            break
        end
    end    
end

% Get clean rows indexes
totRows = 1:size(a,1);
cleanRows = setdiff(totRows , dirtyRows);

% Generate clean table
tClean = t(cleanRows,:);
writetable(tClean , fnameClean , 'WriteVariableNames',false);
