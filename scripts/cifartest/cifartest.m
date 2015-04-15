setenv('LC_ALL','C');
% addp

    % % Visualize some misclassified samples
    % I = (sign(Ypred)~=Yts); 
    % ind = find(I); 
    % nel = numel(ind);
    % for i=1:6 
    %     figure; 
    %     idx = ind(randi(nel));
    %     visualizeExample(Xts(idx,:));
    % endath(genpath('.'));
 
clearAllButBP;
close all;

% Set experimental results relative directory name
% resdir = 'scripts/cifartest/results/';
% mkdir(resdir);

storeFullTrainPerf = 1;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 1;

%% Dataset initialization

% all classes
% ds = Cifar10(50000,10000,'plusMinusOne',1:10);

% 2 classes
% ds = Cifar10(500,1000,'plusMinusOne',[0,1]);

% All possible combinations of 2 classes

classesComb = nchoosek(0:9 , 2);
ncomb = size(classesComb,1);
T = cell(ncomb,5,5);    % Results storage vector
nTrPerClass = 500;
nTePerClass = 1000;

for i = 1:ncomb
    
    classes = classesComb(i,:)
    ds = Cifar10(nTrPerClass,nTePerClass,'plusMinusOne',classes);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%% Tests with ISML2 2015 Library
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Xtr = ds.X(ds.trainIdx,:);
    Ytr = ds.Y(ds.trainIdx,:);
    Xts = ds.X(ds.testIdx,:);
    Yts = ds.Y(ds.testIdx,:);

    %% Knn
    intRegPar = 1:10;
    [l, s, Vm, Vs, Tm, Ts] = holdoutCV('knn', Xtr, Ytr,[], 0.8, 1, intRegPar, []);

    Ypred = kNNClassify(Xtr, Ytr, l, Xts);
    tsErr = calcErr(Ypred, Yts);

    % Display & save results
    T{i,1,1} = cellstr('K Nearest Neighbours');
    display(T{i,1,1});
    display(['Test error = ', num2str(tsErr), ' with optimal k* = ' , num2str(l)]);
    T{i,2,1} = tsErr;
    T{i,3,1} = [];
    T{i,4,1} = l;
    T{i,5,1} = classes;

    %% RLS

    intRegPar = logspace(0,-5,6);
    [l, s, Vm, Vs, Tm, Ts] = holdoutCV('rls', Xtr, Ytr,[], 0.8, 1, intRegPar, []);

    w = regularizedLSTrain(Xtr, Ytr, l);
    Ypred = sign(regularizedLSTest(w, Xts));
    tsErr = calcErr(Ypred, Yts);

    % Display & save results
    T{i,1,2} = cellstr('Regularized Least Squares');
    display(T{i,1,2})
    display(['Test error = ', num2str(tsErr), ' with optimal lambda* = ' , num2str(l)]);
    T{i,2,2} = tsErr;
    T{i,3,2} = [];
    T{i,4,2} = l;
    T{i,5,2} = classes;

    %% KRLS

    kernel = 'gaussian';

    intRegPar = logspace(0,-7,8);
    intKerPar = 4:10;
    [l, s, Vm, Vs, Tm, Ts] = holdoutCV('krls', Xtr, Ytr, kernel , 0.8, 1, intRegPar, intKerPar);

    c = regularizedKernLSTrain(Xtr, Ytr, kernel, s, l);
    Ypred = sign(regularizedKernLSTest(c, Xtr, kernel, s, Xts));
    tsErr = calcErr(Ypred, Yts);
    
    % Display & save results
    T{i,1,3} = cellstr('Kernel Regularized Least Squares');
    display(T{i,1,3})
    display(['Test error = ', num2str(tsErr), ' with optimal lambda* = ' , num2str(l) , ' and optimal sigma* = ' , num2str(s)]);
    T{i,2,3} = tsErr;
    T{i,3,3} = s;
    T{i,4,3} = l;
    T{i,5,3} = classes;

    %% LR

    intRegPar = logspace(0,-5,6);
    [l, s, Vm, Vs, Tm, Ts] = holdoutCV('lr', Xtr, Ytr,[], 0.8, 1, intRegPar, []);

    w = linearLRTrain(Xtr, Ytr, l);
    [Ypred, ppred] = linearLRTest(w, Xts);
    Ypred = sign(Ypred);
    tsErr = calcErr(Ypred, Yts);

    % Display & save results
    T{i,1,4} = cellstr('Logistic Regression');
    display(T{i,1,4})
    display(['Test error = ', num2str(tsErr), ' with optimal lambda* = ' , num2str(l)]);
    T{i,2,4} = tsErr;
    T{i,3,4} = [];
    T{i,4,4} = l;
    T{i,5,4} = classes;

    %% KLR
    kernel = 'gaussian';

    intRegPar = logspace(0,-7,8);
    intKerPar = 4:10;
    [l, s, Vm, Vs, Tm, Ts] = holdoutCV('klr', Xtr, Ytr, kernel , 0.8, 1, intRegPar, intKerPar);

    c = kernLRTrain(Xtr, Ytr, kernel, s, l);
    [Ypred, ppred] =  kernLRTest(c, Xtr, kernel, s, Xts);
    Ypred = sign(Ypred);
    tsErr = calcErr(Ypred, Yts);

    % Display & save results

    T{i,1,5} = cellstr('Kernel Logistic Regression');
    display(T{i,1,5})
    display(['Test error = ', num2str(tsErr), ' with optimal lambda* = ' , num2str(l) , ' and optimal sigma* = ' , num2str(s)]);
    T{i,2,5} = tsErr;
    T{i,3,5} = s;
    T{i,4,5} = l;
    T{i,5,5} = classes;
    
end


%% Incremental Nystrom KRLS
% 
% map = @nystromUniformIncremental;
% 
% numNysParGuesses = 5;
% 
% % alg = incrementalNkrls(map , 500 , 'numNysParGuesses' , numNysParGuesses ,...
% %                         'numMapParGuesses' , 20,  ...
% %                         'numMapParRangeSamples' , 2000,  ...
% %                         'filterParGuesses', 1e-7 , ... %logspace(-7,0,32), ...
% %                         'verbose' , 0 , ...
% %                         'storeFullTrainPerf' , storeFullTrainPerf , ...
% %                         'storeFullValPerf' , storeFullValPerf , ...
% %                         'storeFullTestPerf' , storeFullTestPerf);
% 
% alg = incrementalNkrls(map , 500 , 'numNysParGuesses' , numNysParGuesses ,...
%                         'numMapParGuesses' , 10,  ...
%                         'numMapParRangeSamples' , 1000, ...
%                         'filterParGuesses', 1e-7, ...
%                         'verbose' , 0 , ...
%                         'storeFullTrainPerf' , storeFullTrainPerf , ...
%                         'storeFullValPerf' , storeFullValPerf , ...
%                         'storeFullTestPerf' , storeFullTestPerf);
% 
% expNysInc = experiment(alg , ds , 1 , true , false , 'nm' , resdir , 0);
% expNysInc.run();
% expNysInc.result
% 
% % incrementalnkrls_plots

%% Exact KRLS
% 
% map = @gaussianKernel;
% fil = @tikhonov;
% 
% filterParGuesses = logspace(-7,-3,5);
% 
% alg = krls( map , fil , 'mapParGuesses' , 2:20 , 'filterParGuesses' , filterParGuesses , 'verbose' , 0 , ...
%                         'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , ...
%                         storeFullValPerf , 'storeFullTestPerf' , storeFullTestPerf);
% 
% % alg = krls( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , logspace(-5,0,12) , 'verbose' , 0 , ...
% %                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);
% 
% % alg = krls( map , fil , 'numMapParGuesses' , 10 , 'numFilterParGuesses' , 10 , 'verbose' , 1 , ...
% %                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);
% 
% % alg = krls( map , fil , 'mapParGuesses' , linspace(1,5,10) , 'filterParGuesses' , logspace(-5,1,10) , 'verbose' , 1 , ...
% %                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);
% 
% expKRLS = experiment(alg , ds , 1 , true , true , 'nh' , resdir);
% 
% expKRLS.run();
% expKRLS.result
% 
% krls_plots

%% Display results


for i = 1:9:ncomb
   
    figure
    for j = 1:9
        subplot(3,3,j)
        bar(squeeze(cell2mat(T(i+j-1 , 2, :)))')
        title(['Classes ' , num2str(T{i+j-1 , 5, 1})])
        set(gca,'XTickLabel',{'KNN', 'RLS', 'KRLS', 'LR' , 'KLR'})
        ylabel('Test Error')
    end
end


% 
% 
%% Save stuff
save('./scripts/cifartest/results/5/results.mat')

figsdir = './scripts/cifartest/results/5/';
% mkdir(figsdir);
saveAllFigs