clearAllButBP

%% Dataset initialization

% Load full dataset
ds = USPS;

% Load small dataset
%ds = USPS(1000,500);

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

%% Experiment setup

% ker = gaussianKernel;
% fil = tikhonov;

ker = @gaussianKernel;
fil = @tikhonov;

alg = regls(ker, fil, 10, 10);

exp = experiment('Experiment_USPS_RLS');
exp.run(alg , ds)

exp.result