setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
% ds = MNIST;

% Load small dataset
ds = MNIST(60000,10000,'plusMinusOne');


%% Experiment 1 setup, Gaussian kernel

ker = @gaussianKernel;
fil = @tikhonov;

alg = krls(ker, fil,  1, 25);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);

exp.run();
exp.result

 %% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation
% 
% map = @randomFeaturesGaussian;
% fil = @tikhonov;
% 
% alg = rfrls(map , 1000 , fil,  10 , 10 , 32000);
% 
% exp = experiment(alg , ds , 1 , true , true , 'SERVER' , resdir);
% exp.run();
% 
% exp.result
% % 