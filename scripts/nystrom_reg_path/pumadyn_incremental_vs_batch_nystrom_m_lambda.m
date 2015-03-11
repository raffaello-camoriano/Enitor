setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load dataset
ds = pumadyn(4096,4096, 32 , 'n' , 'h');
% ds = MNIST(30000,10000,'plusMinusOne');
% ds = icubdyn(10000,10000);

%% Set ranges

% Fixed Tikhonov filter lambda parameter guesses
lMin = -5;
lMax = 0;
nLambda = 1;
fixedFilterParGuesses = logspace(lMin,lMax,nLambda);

fixedMapPar = 6;     %Pumadyn
% fixedMapPar = [];     %icubdyn


%% Incremental

map = @nystromUniformIncremental;

% mapType, numKerParRangeSamples, numNysParGuesses,  numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose)
alg = incrementalNkrls(map , 1000 , 20 , 1 , fixedFilterParGuesses , 3000 , fixedMapPar , 0 , 1,1,1);

exp = experiment(alg , ds , 1 , true , true , 'nm' , resdir , 0);
exp.run();

%% Batch

map = @nystromUniform;
fil = @tikhonov;

% mapType, numKerParRangeSamples, filterType, numNysParGuesses , numMapParGuesses , numFilterParGuesses , maxRank , fixedMapPar , fixedFilterPar , verbose , storeFullTrainPerf, storeFullValPerf
alg = nrls(map , 1000 , fil , 1 , 1 , nLambda , 3000, fixedMapPar , fixedFilterParGuesses , 0 , 1,1,1);

exp2 = experiment(alg , ds , 1 , true , true , 'nm' , resdir , 0);
exp2.run();

%% Plot & Display

% Generic recap
display('Summary')
display('Batch:')
exp.result
display('Incremental')
exp2.result

%Time perf
display('Time performances')
display('Batch:')
exp2.result.time
display('Incremental')
exp.result.time

% Accuracy perf
display('Accuracy test set predictions performances')
display('Batch:')
exp2.result.perf
display('Incremental')
exp.result.perf

figure
title('Predictions')
hold on
plot(exp.result.Y)
plot(exp.result.Ypred)
% plot(exp2.result.Ypred)
hold off

figure
title('Weights')
hold on
plot(exp.algo.c)
plot(exp2.algo.c)
hold off

figure
title('Difference between predictions')
hold on
plot(exp.result.Ypred - exp2.result.Ypred)
hold off

figure
title('Difference between weights')
hold on
plot(exp.algo.c - exp2.algo.c)
hold off