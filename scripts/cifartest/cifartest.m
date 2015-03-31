setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = 'scripts/cifartest/results/';
mkdir(resdir);

storeFullTrainPerf = 0;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 1;

%% Dataset initialization

ds = Cifar10(50000,10000,'plusMinusOne',1:10);

% %% Experiment 1 setup, Landweber, Gaussian kernel
% 
% map = @gaussianKernel;
% fil = @gdesc_square_loss;
% maxiter = 20000;
% 
% alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , 1:maxiter , ...
%     'verbose' , 0 , 'storeFullTrainPerf' , storeFullTrainPerf , ...
%     'storeFullValPerf' , storeFullValPerf , ...
%     'storeFullTestPerf' , storeFullTestPerf);
% 
% expLandweber = experiment(alg , ds , 1 , true , true , '' , resdir);
% 
% expLandweber.run();
% expLandweber.result

%% Incremental Nystrom KRLS

map = @nystromUniformIncremental;

numNysParGuesses = 5;

% alg = incrementalNkrls(map , 500 , 'numNysParGuesses' , numNysParGuesses ,...
%                         'numMapParGuesses' , 20,  ...
%                         'numMapParRangeSamples' , 2000,  ...
%                         'filterParGuesses', 1e-7 , ... %logspace(-7,0,32), ...
%                         'verbose' , 0 , ...
%                         'storeFullTrainPerf' , storeFullTrainPerf , ...
%                         'storeFullValPerf' , storeFullValPerf , ...
%                         'storeFullTestPerf' , storeFullTestPerf);
                    
alg = incrementalNkrls(map , 5000 , 'numNysParGuesses' , numNysParGuesses ,...
                        'mapParGuesses' , 1:20,  ...
                        'filterParGuesses', logspace(-9,-5,10), ...
                        'verbose' , 0 , ...
                        'storeFullTrainPerf' , storeFullTrainPerf , ...
                        'storeFullValPerf' , storeFullValPerf , ...
                        'storeFullTestPerf' , storeFullTestPerf);

expNysInc = experiment(alg , ds , 1 , true , false , 'nm' , resdir , 0);
expNysInc.run();
expNysInc.result

incrementalnkrls_plots


%%


% 
% 
%% Save figures
figsdir = resdir;
% mkdir(figsdir);
saveAllFigs