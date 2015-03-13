setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load dataset
ds = pumadyn(4096,4096, 32 , 'n' , 'h');
% ds = MNIST(30000,10000,'plusMinusOne');
% ds = icubdyn(10000,10000);

%% Set ranges

% Fixed Tikhonov filter lambda parameter guesses
lMin = -5;
lMax = 0;
nLambda = 20;
filterParGuesses = logspace(lMin,lMax,nLambda);
numNysParGuesses = 20;
mapParGuesses = 6;     %Pumadyn

%% Incremental

map = @nystromUniformIncremental;

alg = incrementalNkrls(map , 3000 , 'numNysParGuesses' , numNysParGuesses , 'mapParGuesses' , mapParGuesses ,  ...
                        'numMapParRangeSamples' , 1000 , 'filterParGuesses', filterParGuesses , 'verbose' , 0 , ...
                        'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);
                    
expNysInc = experiment(alg , ds , 1 , true , true , 'nm' , resdir , 0);
expNysInc.run();

%% Batch

map = @nystromUniform;
fil = @tikhonov;

alg = nrls(map , fil , 3000 , 'numNysParGuesses' , numNysParGuesses , 'mapParGuesses' , mapParGuesses ,  ...
                        'numMapParRangeSamples' , 1000 , 'filterParGuesses', filterParGuesses , 'verbose' , 0 , ...
                        'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);
                    
expNysBat = experiment(alg , ds , 1 , true , true , 'nm' , resdir , 0);
expNysBat.run();

%% Plot & Display

% Generic recap
display('Summary')
display('Batch:')
expNysInc.result
display('Incremental')
expNysBat.result

%Time perf
display('Time performances')
display('Batch:')
expNysBat.result.time
display('Incremental')
expNysInc.result.time

% Accuracy perf
display('Accuracy test set predictions performances')
display('Batch:')
expNysBat.result.perf
display('Incremental')
expNysInc.result.perf

%%  Incremental update sanity checks
figure
title('Predictions')
hold on
plot(expNysInc.result.Y)
plot(expNysInc.result.Ypred)
plot(expNysBat.result.Ypred)
legend('Ground truth','Incremental','Batch');
hold off

figure
title('Weights')
hold on
plot(expNysInc.algo.c)
plot(expNysBat.algo.c)
legend('Incremental','Batch');
hold off

figure
title('Difference between incremental and batch predictions')
hold on
plot(expNysInc.result.Ypred - expNysBat.result.Ypred)
hold off

figure
title('Difference between weights')
hold on
plot(expNysInc.algo.c - expNysBat.algo.c)
hold off

%% Incremental Nystrom performance (train, val, test)

figure
title('Incremental Nystrom performance')
hold on    
h = surf(expNysInc.algo.trainPerformance);
alpha(h,0.2)
h = surf(expNysInc.algo.valPerformance);
alpha(h,0.2)
h = surf(expNysInc.algo.testPerformance);
alpha(h,0.2)
hold off
legend('Training','Validation','Test');

%% Incremental Nystrom performance (only train)

figure
pcolor(expNysInc.algo.filterParGuesses,expNysInc.algo.nyMapper.rng(1,:),expNysInc.algo.trainPerformance)
title({'Incremental Nystrom performance';'Training Set'})
ylabel('m')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar;
h.Label.String = 'RMSE';

%%
figure
hold on
title({'Incremental Nystrom performance';'Training Set'})
colormap jet
cc=jet(size(expNysInc.algo.nyMapper.rng(1,:),2));    
for i = 1:size(expNysInc.algo.nyMapper.rng(1,:),2)
    plot(expNysInc.algo.filterParGuesses,expNysInc.algo.trainPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(numel(expNysInc.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysInc.algo.nyMapper.rng(1,:) );
h.Label.String = 'm';

%% Incremental Nystrom performance (only val)

figure
imagesc()
pcolor(expNysInc.algo.filterParGuesses,expNysInc.algo.nyMapper.rng(1,:),expNysInc.algo.valPerformance)
title({'Incremental Nystrom performance';'Validation Set'})
ylabel('m')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar;
h.Label.String = 'RMSE';

%%
figure
hold on
title({'Incremental Nystrom performance';'Validation Set'})
colormap jet
cc=jet(size(expNysInc.algo.nyMapper.rng(1,:),2));    
for i = 1:size(expNysInc.algo.nyMapper.rng(1,:),2)
    plot(expNysInc.algo.filterParGuesses,expNysInc.algo.valPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(numel(expNysInc.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysInc.algo.nyMapper.rng(1,:) );
h.Label.String = 'm';

%% Incremental Nystrom performance (only test)

figure
imagesc()
pcolor(expNysInc.algo.filterParGuesses,expNysInc.algo.nyMapper.rng(1,:),expNysInc.algo.testPerformance)
title({'Incremental Nystrom performance';'Test Set'})
ylabel('m')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar;
h.Label.String = 'RMSE';

%%
figure
hold on
title({'Incremental Nystrom performance';'Test Set'})
colormap jet
cc=jet(size(expNysInc.algo.nyMapper.rng(1,:),2));    
for i = 1:size(expNysInc.algo.nyMapper.rng(1,:),2)
    plot(expNysInc.algo.filterParGuesses,expNysInc.algo.testPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(numel(expNysInc.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysInc.algo.nyMapper.rng(1,:) );
h.Label.String = 'm';


%% Incremental Nystrom performance - with repetitions
if numRep > 1
    
    % Median test error surface
    testErrMed = median(expNysInc.algo.testPerformance,3);
    figure
    surf(mRange , lRange , testErrMed)
    title('Incremental Nystrom, median test performance')
    set(gca,'XScale','lin')
    set(gca,'YScale','log')

    % Mean + sd test error surface
    testErrAvg = mean(expNysInc.algo.testPerformance,3);
    testErrSd = std(testErr,1,3);
    figure
    hold on
    title('Incremental Nystrom, mean +- 2\sigma performance')
    surf(mRange , lRange , testErrAvg)
    h = surf(mRange , lRange , testErrAvg + 2*testErrSd);
    alpha(h,0.2)
    h = surf(mRange , lRange , testErrAvg - 2*testErrSd);
    alpha(h,0.2)
    set(gca,'XScale','lin')
    set(gca,'YScale','log')
    hold off
    
end