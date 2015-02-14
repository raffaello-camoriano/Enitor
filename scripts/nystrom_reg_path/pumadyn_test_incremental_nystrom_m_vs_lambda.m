setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = pumadyn;

% Load small dataset
ds = pumadyn(4096,4096, 32 , 'n' , 'h');

%% Experiment setup, Nystrom method with uniform kernel column sampling. Gaussian kernel approximation.

map = @nystromUniformIncremental;
fil = @tikhonov;

% Fixed Tikhonov filter lambda parameter guesses
lMin = -7;
lMax = 0;
nLambda = 10;
filterParGuesses = logspace(lMin,lMax,nLambda);

% mapType, numKerParRangeSamples, numNysParGuesses,  numMapParGuesses , filterParGuesses , maxRank , fixedMapPar , verbose)
alg = incrementalNkrls(map , 1000 , 10 , 1 , filterParGuesses , 3000 , [] , 1 , 1 , 1);

exp = experiment(alg , ds , 1 , true , true , 'nm' , resdir , 1);
exp.run();

exp.result