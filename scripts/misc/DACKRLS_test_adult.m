setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = Adult;

% Load small dataset
ds = Adult(7000,16282,'plusMinusOne');


%% Experiment setup: DACKRLS

% Algorithm init
map = @gaussianKernel;  
fil = @tikhonov;
% mGuesses = [5 , 10 , 15 , 20];
% mGuesses = 5:10:500;
mGuesses = 1:50;
verbose = 0;
storeFullTrainPerf = 1;
storeFullValPerf = 1;
storeFullTestPerf = 1;
mapParGuesses = 10;
filterParGuesses = 0.1;
alg = dackrls(map , fil , mGuesses , 'mapParGuesses' , mapParGuesses , 'filterParGuesses' , filterParGuesses , 'verbose' , verbose , 'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf , 'storeFullTestPerf' , storeFullTestPerf);

% Exp init
exp = experiment(alg , ds , 1 , true , true , '' , resdir);

exp.run();

exp.result

%% Results plotting

figure
plot(mGuesses,cell2mat(exp.algo.trainPerformance));
hold on;
plot(mGuesses,cell2mat(exp.algo.valPerformance));
plot(mGuesses,cell2mat(exp.algo.testPerformance));
hold off;
title('Performances for Varying # of Splits')
legend('Training perf','Validation perf','Test perf')
xlabel('# of Splits')
ylabel('Performance')