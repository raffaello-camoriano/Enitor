setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = 'scripts/gdesc_comparison/padova_plots/';
mkdir(resdir);

%% Dataset initialization

numRep =  10;
storeFullTrainPerf = 1;
storeFullTrainPred = 1;      
storeFullValPerf = 1;
storeFullTestPerf = 1;
storeFullTestPred = 1;      
verbose = 0;
% Load dataset
ds = SynthSinusoid(50 , 50, 1 , 2*pi , 0 , 1 , 0.3);


%% Experiment setup, nu method, Gaussian kernel

map = @gaussianKernel;
fil = @numethod_square_loss;
maxiter = 10000;

alg = kgdesc( map , fil , 'mapParGuesses' , 0.08 , 'filterParGuesses' , 1:maxiter   ,...
            'verbose' , verbose , ...
            'storeFullTrainPerf' , storeFullTrainPerf , ...
            'storeFullTrainPred' , storeFullTrainPred , ...
            'storeFullValPerf' , storeFullValPerf , ...
            'storeFullTestPerf' , storeFullTestPerf , ...
            'storeFullTestPred' , storeFullTestPred);

expNuMethod = experiment(alg , ds , 1 , true , true , '' , resdir);

expNuMethod.run();
expNuMethod.result

numethod_plots

%% plots

% Plot differently regularized solutions

%% Test set
% figure
% hold on
% scatter(ds.X(ds.testIdx),expNuMethod.result.Y)
% [B,I] = sort(ds.X(ds.testIdx));
% tmp = squeeze(expNuMethod.algo.testPred(1,1,:));
% plot(B,tmp(I))
% hold off
% 
% figure
% hold on
% scatter(ds.X(ds.testIdx),expNuMethod.result.Y)
% [B,I] = sort(ds.X(ds.testIdx));
% tmp = squeeze(expNuMethod.algo.testPred(1,floor(expNuMethod.result.filterParStar/2),:));
% plot(B,tmp(I))
% hold off
% 
% figure
% hold on
% scatter(ds.X(ds.testIdx),expNuMethod.result.Y)
% [B,I] = sort(ds.X(ds.testIdx));
% tmp = squeeze(expNuMethod.algo.testPred(1,expNuMethod.result.filterParStar,:));
% plot(B,tmp(I))
% hold off
% 
% figure
% hold on
% scatter(ds.X(ds.testIdx),expNuMethod.result.Y)
% [B,I] = sort(ds.X(ds.testIdx));
% tmp = squeeze(expNuMethod.algo.testPred(1,end,:));
% plot(B,tmp(I))
% hold off


%% Training set


figure
hold on
Xtr = ds.X(ds.trainIdx);
Ytr = ds.Y(ds.trainIdx);
scatter(Xtr(alg.trainIdx),Ytr(alg.trainIdx))
iteration = 1;
title({'Fitting on the training set';['Iteration #' , num2str(iteration)]})
[B,I] = sort(Xtr(alg.trainIdx));
tmp = squeeze(expNuMethod.algo.trainPred(1,iteration,:));
plot(B,tmp(I))
hold off
% M(1) = getframe;
M(1) = getframe(gca);


figure
hold on
Xtr = ds.X(ds.trainIdx);
Ytr = ds.Y(ds.trainIdx);
scatter(Xtr(alg.trainIdx),Ytr(alg.trainIdx))
iteration = floor(alg.filterParStar/3);
title({'Fitting on the training set';['Iteration #' , num2str(iteration)]})
[B,I] = sort(Xtr(alg.trainIdx));
tmp = squeeze(expNuMethod.algo.trainPred(1,iteration,:));
plot(B,tmp(I))
hold off
% M(2) = getframe;
M(2) = getframe(gca);

figure
hold on
Xtr = ds.X(ds.trainIdx);
Ytr = ds.Y(ds.trainIdx);
scatter(Xtr(alg.trainIdx),Ytr(alg.trainIdx))
iteration = alg.filterParStar;
% title({'Fitting on the training set';['Iteration #' , num2str(iteration)];'Optimal case'})
title({'Fitting on the training set';['Iteration #' , num2str(iteration)]})
[B,I] = sort(Xtr(alg.trainIdx));
tmp = squeeze(expNuMethod.algo.trainPred(1,iteration,:));
plot(B,tmp(I))
hold off
% M(3) = getframe;
M(3) = getframe(gca);


figure
hold on
Xtr = ds.X(ds.trainIdx);
Ytr = ds.Y(ds.trainIdx);
scatter(Xtr(alg.trainIdx),Ytr(alg.trainIdx))
iteration = 5000;
title({'Fitting on the training set';['Iteration #' , num2str(iteration)]})
[B,I] = sort(Xtr(alg.trainIdx));
tmp = squeeze(expNuMethod.algo.trainPred(1,iteration,:));
plot(B,tmp(I))
hold off
% M(4) = getframe;
M(4) = getframe(gca);


figure
hold on
Xtr = ds.X(ds.trainIdx);
Ytr = ds.Y(ds.trainIdx);
scatter(Xtr(alg.trainIdx),Ytr(alg.trainIdx))
iteration = maxiter;
title({'Fitting on the training set';['Iteration #' , num2str(iteration)]})
[B,I] = sort(Xtr(alg.trainIdx));
tmp = squeeze(expNuMethod.algo.trainPred(1,iteration,:));
plot(B,tmp(I))
hold off
% M(5) = getframe;
M(5) = getframe(gca);

figure
movie(M,1,3)

movie2gif(M, 'test.gif' , 'DelayTime' , 1.5 , 'LoopCount' , Inf)