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
ds = MNIST(20000,10000,'plusMinusOne');

% dataset.n
% dataset.nTr
% dataset.nTe
% dataset.d
% dataset.t
% 
% dataset.X
% dataset.Y
% 
% dataset.trainIdx
% dataset.testIdx

% Shuffled training set indexes
% dataset.shuffledTrainIdx      
% dataset.shuffleTrainIdx();
% dataset.shuffledTrainIdx

% Perf
% Y = (1:10)';
% Ypred = (10:-1:1)';
% dataset.performanceMeasure(Y , Ypred)l;

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

alg = rfrls(map , 1000 , fil,  50 , 50 , 20000);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);
exp.run();

exp.result
