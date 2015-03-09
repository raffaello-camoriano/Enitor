setenv('LC_ALL','C');
addpath(genpath('.'));

clearAllButBP

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = Higgs;

% Load small dataset
ds = Higgs(5000,5000);

warning('Higgs performance measure set to Error Rate.AUC to be implemented');
warning('Higgs contains missing values, which are not dealt with at the moment. Results may be invalidated!');
warning('The dataset is unbalanced, proper classification methods have to be implemented');

%% Experiment 1 setup, Gaussian kernel

ker = @gaussianKernel;
fil = @tikhonov;

alg = krls(ker, fil,  5, 5);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);

exp.run();

exp.result

%% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation
% 
map = @randomFeaturesGaussian;
fil = @tikhonov;

alg = rfrls(map , 1000 , fil,  2 , 1);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);
exp.run();

exp.result