setenv('LC_ALL','C');
% addpath(genpath('.'));
% addpath(genpath('/home/iit.local/rcamoriano/repos/Enitor'));
 
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = '';
mkdir(resdir);

%% Initialization

numRep = 1;
storeFullTrainPerf = 0;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 0;
saveResult = 0;

%% Storage vars init
    
% Training time
KRLS_cumulative_training_time= zeros(numRep,1);
DACKRLS_cumulative_training_time= zeros(numRep,1);
Landweber_cumulative_training_time= zeros(numRep,1);
NuMethod_cumulative_training_time= zeros(numRep,1);
NysInc_cumulative_training_time= zeros(numRep,1);
RFInc_cumulative_training_time= zeros(numRep,1);
gdesc_kernel_hinge_loss_cumulative_training_time= zeros(numRep,1);
FFRLS_cumulative_training_time = zeros(numRep,1);

% Testing time
KRLS_cumulative_testing_time= zeros(numRep,1);
DACKRLS_cumulative_testing_time= zeros(numRep,1);
Landweber_cumulative_testing_time= zeros(numRep,1);
NuMethod_cumulative_testing_time= zeros(numRep,1);
NysInc_cumulative_testing_time= zeros(numRep,1);
RFInc_cumulative_testing_time= zeros(numRep,1);
gdesc_kernel_hinge_loss_cumulative_testing_time= zeros(numRep,1);
FFRLS_cumulative_testing_time = zeros(numRep,1);


% Test performance
KRLS_cumulative_test_perf = zeros(numRep,1);
DACKRLS_cumulative_test_perf= zeros(numRep,1);
Landweber_cumulative_test_perf= zeros(numRep,1);
NuMethod_cumulative_test_perf= zeros(numRep,1);
NysInc_cumulative_test_perf = zeros(numRep,1);
RFInc_cumulative_test_perf = zeros(numRep,1);
gdesc_kernel_hinge_loss_cumulative_test_perf= zeros(numRep,1);
FFRLS_cumulative_test_perf = zeros(numRep,1);

% incremental nystrom storage vars

nysTrainTime = [];
nysTestPerformance = [];
nysValPerformance  = [];

for k = 1:numRep

    display([ 'Repetition #', num2str(k)])
     
    % Load dataset
   ds = Adult(32000,16282,'plusMinusOne');
    

    %% Incremental Nystrom KRLS

%     map = @nystromUniformIncremental_unstable;
    map = @nystromUniformIncremental;

    numNysParGuesses = 20;
%     filterParGuesses = logspace(0,-9,10);
    filterParGuesses = logspace(0,-9,10);
    mapParGuesses = 6.6;

    alg = incrementalNkrls(map , 3000 , ...
                            'minRank' , 1 , ...
                            'numNysParGuesses' , numNysParGuesses ,...
                            'mapParGuesses' , mapParGuesses ,  ...
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf);
                        
%     alg = incrementalNkrls(map , 30000 , ...
%                             'minRank' , 100 , ...
%                             'numNysParGuesses' , numNysParGuesses ,...
%                             'numMapParGuesses' , 10 ,  ...
%                             'numMapParRangeSamples' , 10000 ,  ...
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf);

%     alg.mapParStar = [30000 , 7];
%     alg.filterParStar = 1e-8;
% 
%     
%     tic
%     alg.justTrain(ds.X(ds.trainIdx,:) , ds.Y(ds.trainIdx,:));
%     trainTime = toc;
%     
%     tic
%     YtePred = alg.test(ds.X(ds.testIdx,:));   
%     testTime = toc;
%     
%     perf = abs(ds.performanceMeasure( ds.Y(ds.testIdx,:) , YtePred , ds.testIdx));

    expNysInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
    expNysInc.run();
    
    if numRep > 1
        nysValPerformance = [nysValPerformance ; expNysInc.algo.valPerformance'];
    end
    
    
%     expNysInc.result
% 
%     NysInc_cumulative_training_time(k) = expNysInc.time.train;
%     NysInc_cumulative_testing_time(k) = expNysInc.time.test;
%     NysInc_cumulative_test_perf(k) = expNysInc.result.perf;

    % incrementalnkrls_plots


    %% Batch Nystrom KRLS

    map = @nystromUniform;
    filter = @tikhonov;
    
    mapParGuesses = 6.6;
%     mapParGuesses = linspace(1,1.5,10);
    filterParGuesses = 1e-9;

    alg = nrls(map , filter , 5000 , ...
                            'mapParGuesses' , mapParGuesses ,  ... 
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf );

%     alg.mapParStar = [ 0 , 5.7552];
%     alg.filterParStar = 1e-9;
    
%     tic
%     alg.justTrain(ds.X(ds.trainIdx,:) , ds.Y(ds.trainIdx));
%     trTime = toc;
%     
%     YtePred = alg.test(ds.X(ds.testIdx,:));   
%       
%     perf = abs(ds.performanceMeasure( ds.Y(ds.testIdx,:) , YtePred , ds.testIdx));
%     nysTrainTime = [nysTrainTime ; trTime];
%     nysTestPerformance = [nysTestPerformance ; perf'];

    expNysBat = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
    expNysBat.run();
    expNysBat.result

%     nysTrainTime = [nysTrainTime ; expNysInc.algo.trainTime'];
    nysTestPerformance = [nysTestPerformance ; expNysBat.algo.testPerformance'];

    NysBat_cumulative_training_time(k) = expNysBat.time.train;
    NysBat_cumulative_testing_time(k) = expNysBat.time.test;
    NysBat_cumulative_test_perf(k) = expNysBat.result.perf;
    
end

% save('wspace_plot1_adult_laptop.mat' , '-v7.3');


%% Plot 1 nips15

%% Incremental Nystrom performance (only val)

figure
% imagesc()
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
