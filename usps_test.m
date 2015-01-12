setenv('LC_ALL','C')
addpath(genpath('.'));

clearAllButBP

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
ds = USPS(7291,2007,'plusMinusOne');

% Load small dataset
%ds = USPS(1000,1000,'plusMinusOne');

%% Experiment 1 setup, Gaussian kernel

ker = @gaussianKernel;
fil = @tikhonov;

alg = krls(ker, fil,  25, 25);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);

exp.run();

exp.result

%% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation

% map = @randomFeaturesGaussian;
% fil = @tikhonov;
% 
% alg = rfrls(map , 1000 , fil,  5, 5, 8000);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% exp.run();
% 
% exp.result

%% Experiment 3 setup, Nystrom method with uniform kernel column sampling. Gaussian kernel approximation.

map = @nystromUniform;
fil = @tikhonov;

alg = nrls(map , 1000 , fil,  25 , 25 , 2000);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);
exp.run();

exp.result
exp.result.mapParStar
exp.result.filterParStar