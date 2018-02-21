setenv('LC_ALL','C');
addpath(genpath('.'));

clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = '';
mkdir(resdir);

%% Initialization

numRep = 10;
storeFullTrainPerf = 0;
storeFullValPerf = 1;
storeFullTestPerf = 0;
storeFullTrainTime = 1;
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
RFBat_cumulative_training_time= zeros(numRep,1);
gdesc_kernel_hinge_loss_cumulative_training_time= zeros(numRep,1);
FFRLS_cumulative_training_time = zeros(numRep,1);

% Testing time
KRLS_cumulative_testing_time= zeros(numRep,1);
DACKRLS_cumulative_testing_time= zeros(numRep,1);
Landweber_cumulative_testing_time= zeros(numRep,1);
NuMethod_cumulative_testing_time= zeros(numRep,1);
NysInc_cumulative_testing_time= zeros(numRep,1);
RFInc_cumulative_testing_time= zeros(numRep,1);
RFBat_cumulative_testing_time= zeros(numRep,1);
gdesc_kernel_hinge_loss_cumulative_testing_time= zeros(numRep,1);
FFRLS_cumulative_testing_time = zeros(numRep,1);


% Test performance
KRLS_cumulative_test_perf = zeros(numRep,1);
DACKRLS_cumulative_test_perf= zeros(numRep,1);
Landweber_cumulative_test_perf= zeros(numRep,1);
NuMethod_cumulative_test_perf= zeros(numRep,1);
NysInc_cumulative_test_perf = zeros(numRep,1);
RFInc_cumulative_test_perf = zeros(numRep,1);
RFBat_cumulative_test_perf = zeros(numRep,1);
gdesc_kernel_hinge_loss_cumulative_test_perf= zeros(numRep,1);
FFRLS_cumulative_test_perf = zeros(numRep,1);

% incremental nystrom storage vars

nysTrainTime = [];
nysTestPerformance = [];

for k = 1:numRep

    display([ 'Repetition #', num2str(k)])
     
    % Load dataset
    ds = cpuSmall(6554,1638);
    
%     %% Incremental Nystrom KRLS
% 
%     map = @nystromUniformIncremental;
% 
%     numNysParGuesses = 10;
%     filterParGuesses = 2^(-4);
%     mapParGuesses = 0.5;
% 
%     alg = incrementalNkrls(map , 512 , ...
%                             'minRank' , 10 , ...
%                             'numNysParGuesses' , numNysParGuesses ,...
%                             'mapParGuesses' , mapParGuesses ,  ... 
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf , ...
%                             'storeFullTrainTime' , storeFullTrainTime);
% 
% %     alg = incrementalNkrls(map , 1000 , 'numNysParGuesses' , numNysParGuesses ,...
% %                             'numMapParGuesses' , 10 ,  ...
% %                             'numMapParRangeSamples' , 5000 ,  ...
% %                             'filterParGuesses', filterParGuesses , ...
% %                             'verbose' , 0 , ...
% %                             'storeFullTrainPerf' , storeFullTrainPerf , ...
% %                             'storeFullValPerf' , storeFullValPerf , ...
% %                             'storeFullTestPerf' , storeFullTestPerf);
% 
% %     alg.mapParStar = [400 , 5];
% %     alg.filterParStar = 1e-8;
% %     alg.justTrain(ds.X(ds.trainIdx,:) , ds.Y(ds.trainIdx));
% % 
% %     YtePred = alg.test(ds.X(ds.testIdx,:));   
% %       
% %     perf = abs(ds.performanceMeasure( ds.Y(ds.testIdx,:) , YtePred , ds.testIdx));
% 
%     expNysInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
%     expNysInc.run();
%     expNysInc.result
% 
%     nysTrainTime = [nysTrainTime ; expNysInc.algo.trainTime'];
%     nysTestPerformance = [nysTestPerformance ; expNysInc.algo.testPerformance'];
% 
%     NysInc_cumulative_training_time(k) = expNysInc.time.train;
%     NysInc_cumulative_testing_time(k) = expNysInc.time.test;
%     NysInc_cumulative_test_perf(k) = expNysInc.result.perf;

    % incrementalnkrls_plots


    %% Batch Nystrom KRLS

    map = @nystromUniform;
    filter = @tikhonov;
    
    mapParGuesses = 10;
    numMapParGuesses = 10;
    numMapParRangeSamples = 2000;
    filterParGuesses = 1e-9;

%     alg = nrls(map , filter , 300 , ...
%                             'mapParGuesses' , mapParGuesses ,  ... 
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf );

    alg = nrls(map , filter , 300 , ...
                            'numMapParGuesses' , mapParGuesses ,  ... 
                            'numMapParRangeSamples' , mapParGuesses ,  ... 
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

%% Plots

if numRep == 1
   
    % Plot timing + perf

    figure
    hold on
    plot(expNysInc.algo.trainTime , expNysInc.algo.testPerformance , 'Marker' , 'diamond')
%     plot(expRFInc.algo.trainTime , expRFInc.algo.testPerformance , 'Marker' , 'square')
%     plot(rfTrainTime , rfTestPerformance , 'Marker' , 'square')
%     boxplot(nysTestPerformance ,median(nysTrainTime) ,  'plotstyle','compact')
    ylabeboxplotl('Test RMSE')
    xlabel('Training time (s)')
    legend('Inc Nys','RKS')
    
end

if numRep > 1
    
    % Plot timing + perf

    figure
    hold on
%     plot(expNysInc.algo.trainTime , expNysInc.algo.testPerformance , 'Marker' , 'diamond')
%     plot(expRFInc.algo.trainTime , expRFInc.algo.testPerformance , 'Marker' , 'square')
%     plot(rfTrainTime , rfTestPerformance , 'Marker' , 'square')
    boxplot(nysTestPerformance , median(nysTrainTime) ,  'plotstyle' , 'compact' , 'positions' , median(nysTrainTime))
    ylabel('Test RMSE')
    xlabel('Training time (s)')
    legend('Inc Nys','RKS')
    
    
end


% 
% %%
% % 
% % plots
% % 
% % %% Save figures
% figsdir = resdir;
% % % mkdir(figsdir);
% saveAllFigs