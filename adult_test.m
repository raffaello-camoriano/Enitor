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
ds = Adult(5000,16282,'plusMinusOne');

%% Experiment 1 setup, Gaussian kernel

% ker = @gaussianKernel;
% fil = @tikhonov;
% 
% alg = krls(ker, fil,  10, 10);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% 
% exp.run();
% exp.result

%% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation

% map = @randomFeaturesGaussian;
% fil = @tikhonov;
% 
% alg = rfrls(map , 1000 , fil,  10 , 10 , 5000);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% exp.run();
% 
% exp.result

%% Experiment 3 setup, Nystrom method with uniform kernel column sampling. Gaussian kernel approximation.

map = @nystromUniform;
fil = @tikhonov;

alg = nrls(map , 1000 , fil,  5 , 5 , 3000);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);
exp.run();

exp.result