setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = MNIST;

% Load small dataset
ds = MNIST(7000,10000,'plusMinusOne');

%% Experiment 1 setup, Gaussian kernel

% % ker = @gaussianKernel;
% ker = @laplaceKernel;
% fil = @tikhonov;
% 
% alg = krls(ker, fil,  1, 25);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% 
% exp.run();
% exp.result

 %% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation

% map = @randomFeaturesGaussian;
% fil = @tikhonov;
% 
% alg = rfrls(map , 1000 , fil,  5 , 5);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% exp.run();
% 
% exp.result

%% Experiment 3 setup, Nystrom method with uniform kernel column sampling. Gaussian kernel approximation.
% 
map = @nystromUniform;
fil = @tikhonov;

alg = nrls(map , 1000 , fil,  25 , 25 , 1000 , [] , [] , 0);

exp = experiment(alg , ds , 1 , true , true , '' , resdir , 0);
exp.run();

exp.result
exp.result.mapParStar
exp.result.filterParStar