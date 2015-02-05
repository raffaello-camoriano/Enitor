setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
ds = Sido;

% Load small dataset
%ds = Sido(500,500);

%% Experiment 1 setup, Gaussian kernel

ker = @gaussianKernel;
fil = @tikhonov;

alg = krls(ker, fil,  30, 30 , 0);

exp = experiment(alg , ds , 1 , true , true , '' , resdir , 0);

exp.run();

exp.result

%% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation
% 
% map = @randomFeaturesGaussian;
% fil = @tikhonov;
% 
% alg = rfrls(map , 1000 , fil,  2 , 1);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% exp.run();
% 
% exp.result