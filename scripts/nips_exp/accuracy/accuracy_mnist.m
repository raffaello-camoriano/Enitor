%% Author: Raffaello Camoriano
% In this experiment, we compare the accuracy of the following methods:
%
% - Exact KRLS
% - Divide & conquer KRLS
% - Batch Nystrom KRLS
% - Incremental Nystrom KRLS
% - Random Features KRLS
%
% The benchmark dataset for this experiment is:
% - MNIST

%%

setenv('LC_ALL','C');
% addpath(genpath('.'));
 
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = 'scripts/nips_exp/accuracy/results';
mkdir(resdir);
addpath(genpath('.'));

% Number of repetitions
numRep = 1;

%% Load dataset

ds = MNIST(500,1000,'plusMinusOne');

%% Exact KRLS

map = @gaussianKernel;
fil = @tikhonov;

alg = krls( map , fil , 'numMapParGuesses' , 10 , 'filterParGuesses' , logspace(-5,0,6) , 'verbose' , 0 , ...
                        'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);

% alg = krls( map , fil , 'numMapParGuesses' , 10 , 'numFilterParGuesses' , 10 , 'verbose' , 1 , ...
%                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);

% alg = krls( map , fil , 'mapParGuesses' , linspace(1,5,10) , 'filterParGuesses' , logspace(-5,1,10) , 'verbose' , 1 , ...
%                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);

expKRLS = experiment(alg , ds , 1 , true , true , 'nh' , resdir);

expKRLS.run();
expKRLS.result

krls_plots

%% Divide & Conquer KRLS

% Algorithm init
map = @gaussianKernel;  
fil = @tikhonov;
% mGuesses = [10 50 100];
% mGuesses = [10 50];
mGuesses = [2 3];
verbose = 0;
storeFullTrainPerf = 1;
storeFullValPerf = 1;
storeFullTestPerf = 1;
mapParGuesses = expKRLS.algo.mapParStar;
mapParStarIdx = find(expKRLS.algo.mapParGuesses==mapParGuesses);
filterParGuesses = expKRLS.algo.filterParGuessesStorage(mapParStarIdx,:);
alg = dackrls(map , fil , mGuesses , 'mapParGuesses' , mapParGuesses , 'filterParGuesses' , filterParGuesses ,...
    'verbose' , verbose , 'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf ,...
    'storeFullTestPerf' , storeFullTestPerf);

% Exp init
expDACKRLS = experiment(alg , ds , 1 , true , true , '' , resdir);

expDACKRLS.run();
expDACKRLS.result

dackrls_plots

%% Batch Nystrom KRLS

% map = @nystromUniform;
% fil = @tikhonov;
% 
% % alg = nrls(map , fil , 3000 , 'numNysParGuesses' , numNysParGuesses , 'mapParGuesses' , mapParGuesses ,  ...
% %                         'numMapParRangeSamples' , 1000 , 'filterParGuesses', filterParGuesses , 'verbose' , 0 , ...
% %                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);
%      
% numNysParGuesses = 20;
% 
% alg = nrls(map , fil , 800 , 'numNysParGuesses' , numNysParGuesses , 'mapParGuesses' , mapParGuesses ,  ...
%                         'filterParGuesses', filterParGuesses , 'verbose' , 0 , ...
%                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);
%                     
% expNysBat = experiment(alg , ds , 1 , true , true , 'nm' , resdir , 0);
% expNysBat.run();
% expNysBat.result
% 
% batchnkrls_plots

%% Incremental Nystrom KRLS

map = @nystromUniformIncremental;

numNysParGuesses = 20;

alg = incrementalNkrls(map , 300 , 'numNysParGuesses' , numNysParGuesses , 'mapParGuesses' , mapParGuesses ,  ...
                        'filterParGuesses', filterParGuesses , 'verbose' , 0 , ...
                        'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);
                    
expNysInc = experiment(alg , ds , 1 , true , true , 'nm' , resdir , 0);
expNysInc.run();
expNysInc.result

incrementalnkrls_plots

%% Random Features KRLS

%% Plot timing

figure
trainingTimes = [ expKRLS.result.time.train , expDACKRLS.result.time.train , expNysInc.result.time.train ];
bar(trainingTimes)
set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'incNKRLS'})
title('Training & Model Selection Times')
ylabel('Time (s)')

figure
trainingTimes = [ expKRLS.result.time.test , expDACKRLS.result.time.test , expNysInc.result.time.test ];
bar(trainingTimes)
set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'incNKRLS'})
title('Testing Times')
ylabel('Time (s)')

%% Plot best test performances

figure
testPerf = [ expKRLS.result.perf , expDACKRLS.result.perf , expNysInc.result.perf ];
bar(testPerf)
set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'incNKRLS'})
title('Best test performance')
ylabel('Relative Classification Error')

%% Save figures
figsdir = 'scripts/nips_exp/accuracy/results/figures/mnist';
mkdir(figsdir);
saveAllFigs