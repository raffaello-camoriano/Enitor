setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = icubdyn;

% Load small dataset
ds = icubdyn(10000,10000);

%% Experiment 1 setup, Gaussian kernel

% ker = @gaussianKernel;
% fil = @tikhonov;
% 
% alg = krls(ker, fil,  5, 5);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% 
% exp.run();
% exp.result

 %% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation

map = @randomFeaturesGaussian;
fil = @tikhonov;

alg = rfrls(map , 1000 , fil,  5 , 5);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);
exp.run();

exp.result