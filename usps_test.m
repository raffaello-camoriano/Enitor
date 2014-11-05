clearAllButBP

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = USPS;

% Load small dataset
ds = USPS(1000,500,'plusOneMinusBalanced');

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

ker = @gaussianKernel;
fil = @tikhonov;

alg = kregls(ker, fil,  5, 5);

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