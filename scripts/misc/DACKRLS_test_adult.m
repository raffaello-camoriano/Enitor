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
mGuesses = [5 , 10 , 15 , 20];
verbose = 1;
storeFullTrainPerf = 1;
storeFullValPerf = 1;
storeFullTestPerf = 1;
mapParGuesses = 1;
filterParGuesses = 0.1;
alg = dackrls(map , fil , mGuesses , 'mapParGuesses' , mapParGuesses , 'filterParGuesses' , filterParGuesses , 'verbose' , verbose , 'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf , 'storeFullTestPerf' , storeFullTestPerf);

% Exp init
exp = experiment(alg , ds , 1 , true , true , '' , resdir);

exp.run();

exp.result
exp.result.mapParStar
exp.result.filterParStar