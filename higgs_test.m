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
% numRF = 500;
% mappingType = 'gaussian';
% 
% RFmapper = randomFeaturesMapper( ds.d , numRF , mappingType);
% 
% ds
% fil = @tikhonov;
% 
% alg = regls(fil, 20);
% 
% exp = experiment('Experiment_USPS_RFRLS');
% exp.run(alg , ds)
% 
% exp.result