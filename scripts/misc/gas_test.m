setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%%%%%%%%%%%%%%%%%%%%%%
%  Batch 2
%%%%%%%%%%%%%%%%%%%%%%

%% Dataset initialization

% Load full dataset
%ds = Gas;

% Load small dataset
ds = Gas(1000,244,'plusMinusOne' , 2);
% ds = Gas(1000,586,'plusMinusOne' , 3);

%% Experiment 1 setup, Gaussian kernel

ker = @gaussianKernel;
fil = @tikhonov;

alg = krls(ker, fil,  25, 25);

exp = experiment(alg , ds , 1 , true , true , '2' , resdir);

exp.run();
exp.result

 %% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation

% map = @randomFeaturesGaussian;
% fil = @tikhonov;
% 
% alg = rfrls(map , 1000 , fil,  10 , 10 , 1000);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% exp.run();
% 
% exp.result

%% Experiment 3 setup, Nystrom method with uniform kernel column sampling. Gaussian kernel approximation.

map = @nystromUniform;
fil = @tikhonov;

alg = nrls(map , 500 , fil,  25 , 25 , 500);

exp = experiment(alg , ds , 1 , true , true , '2' , resdir);
exp.run();

exp.result

%%

%%%%%%%%%%%%%%%%%%%%%%
%  Batch 3
%%%%%%%%%%%%%%%%%%%%%%

%% Dataset initialization

% Load full dataset
%ds = Gas;

% Load small dataset
ds = Gas(1000,586,'plusMinusOne' , 3);

%% Experiment 1 setup, Gaussian kernel

ker = @gaussianKernel;
fil = @tikhonov;

alg = krls(ker, fil,  25, 25);

exp = experiment(alg , ds , 1 , true , true , '3' , resdir);

exp.run();
exp.result

 %% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation

% map = @randomFeaturesGaussian;
% fil = @tikhonov;
% 
% alg = rfrls(map , 1000 , fil,  10 , 10 , 1000);
% 
% exp = experiment(alg , ds , 1 , true , true , '' , resdir);
% exp.run();
% 
% exp.result

%% Experiment 3 setup, Nystrom method with uniform kernel column sampling. Gaussian kernel approximation.

map = @nystromUniform;
fil = @tikhonov;

alg = nrls(map , 500 , fil,  25 , 25 , 500);

exp = experiment(alg , ds , 1 , true , true , '3' , resdir);
exp.run();

exp.result