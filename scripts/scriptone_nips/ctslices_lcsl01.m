
setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = '';
mkdir(resdir);

%% Initialization

numRep = 10;
% numRep = 10;
storeFullTrainPerf = 0;
storeFullValPerf = 0;
storeFullTestPerf = 0;
storeFullTrainTime = 0;
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
    ds = CTslices(42800,10700);


    %% Incremental Nystrom KRLS

    map = @nystromUniformIncremental;

    numNysParGuesses = 10;
    filterParGuesses = 1e-7;
    mapParGuesses = 5.2613;
%     mapParGuesses = linspace(1,10,10);
    
    alg = incrementalNkrls(map , 2048 , ...
                            'minRank' , 10 , ...
                            'numNysParGuesses' , numNysParGuesses ,...
                            'mapParGuesses' , mapParGuesses ,  ... 
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf , ...
                            'storeFullTrainTime' , storeFullTrainTime);
    
%     alg = incrementalNkrls(map , 2048 , ...
%                             'minRank' , 10 , ...
%                             'numNysParGuesses' , numNysParGuesses ,...
%                             'numMapParGuesses' , 10 ,  ... 
%                             'numMapParRangeSamples' , 5000 , ...
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf , ...
%                             'storeFullTrainTime' , storeFullTrainTime);

    alg.mapParStar = [ 0 , mapParGuesses];
    alg.filterParStar = filterParGuesses;
    tic
    alg.justTrain(ds.X(ds.trainIdx,:) , ds.Y(ds.trainIdx));
    trTime = toc;
    
    YtePred = alg.test(ds.X(ds.testIdx,:));   
      
    perf = abs(ds.performanceMeasure( ds.Y(ds.testIdx,:) , YtePred , ds.testIdx))
    nysTrainTime = [nysTrainTime ; trTime];
    nysTestPerformance = [nysTestPerformance ; perf'];

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

    
%     %% Batch Nystrom KRLS
% 
%     map = @nystromUniform;
%     filter = @tikhonov;
%     
%     mapParGuesses = 5.2613;
% %     mapParGuesses = linspace(1,1.5,10);
%     filterParGuesses = 1e-15;
% 
%     alg = nrls(map , filter , 2048 , ...
%                             'mapParGuesses' , mapParGuesses ,  ... 
%                             'filterParGuesses', filterParGuesses , ...
%                             'numNysParGuesses' , numNysParGuesses ,...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf );
% 
% %     alg.mapParStar = [ 0 , 5.7552];
% %     alg.filterParStar = 1e-9;
%     
% %     tic
% %     alg.justTrain(ds.X(ds.trainIdx,:) , ds.Y(ds.trainIdx));
% %     trTime = toc;
% %     
% %     YtePred = alg.test(ds.X(ds.testIdx,:));   
% %       
% %     perf = abs(ds.performanceMeasure( ds.Y(ds.testIdx,:) , YtePred , ds.testIdx));
% %     nysTrainTime = [nysTrainTime ; trTime];
% %     nysTestPerformance = [nysTestPerformance ; perf'];
% 
%     expNysBat = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
%     expNysBat.run();
%     expNysBat.result
% 
% %     nysTrainTime = [nysTrainTime ; expNysInc.algo.trainTime'];
%     nysTestPerformance = [nysTestPerformance ; expNysBat.algo.testPerformance'];
% 
%     NysInc_cumulative_training_time(k) = expNysBat.time.train;
%     NysInc_cumulative_testing_time(k) = expNysBat.time.test;
%     NysInc_cumulative_test_perf(k) = expNysBat.result.perf;
% 
%     incrementalnkrls_plots

    

     %% Batch Random Features RLS

%     map = @randomFeaturesGaussian;
%     fil = @tikhonov;
% 
%     
%         filterParGuesses = 1e-7;
% %         filterParGuesses = 2^(-5);
% 
%     %     mapParGuesses = linspace(0.1 , 10 , 10);
% %         mapParGuesses = 0.561;
%         mapParGuesses = 4;
% 
%     alg = rfrls(map , fil , 1000 , ...
%                                 'mapParGuesses' , mapParGuesses ,  ...
%                                 'filterParGuesses', filterParGuesses , ...
%                                 'verbose' , 0 , ...
%                                 'storeFullTrainPerf' , storeFullTrainPerf , ...
%                                 'storeFullValPerf' , storeFullValPerf , ...
%                                 'storeFullTestPerf' , storeFullTestPerf);
% 
%     expRFBat = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
%     expRFBat.run();
%     expRFBat.result
% 
%     RFBat_cumulative_training_time(k) = expRFBat.time.train;
%     RFBat_cumulative_testing_time(k) = expRFBat.time.test;
%     RFBat_cumulative_test_perf(k) = expRFBat.result.perf;

    % incrementalrfrls_plots

    %% Incremental Random Features RLS

%     map = @randomFeaturesGaussianIncremental;
% 
%     numRFParGuesses = 10;
%     
%     filterParGuesses = 1e-7;
% %     filterParGuesses = 100;
% %     filterParGuesses = 2^(-5);
% %     filterParGuesses = logspace(0,-8,9);
% 
% %     mapParGuesses = linspace(0.1 , 10 , 10);
% %     mapParGuesses = 0.561;
%     mapParGuesses = 4;
%     
%     alg = incrementalrfrls(map , 1000 , ...
%                             'minRank' , 100 , ...
%                             'numRFParGuesses' , numRFParGuesses ,...
%                             'mapParGuesses' , mapParGuesses ,  ...
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf, ...
%                             'storeFullTrainTime' , storeFullTrainTime);
% 
%     expRFInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
%     expRFInc.run();
%     expRFInc.result


%     map = @randomFeaturesGaussianIncremental;
% 
%     numRFParGuesses = 1;
%     
%     filterParGuesses = 1e-7;
% %     filterParGuesses = 100;
% %     filterParGuesses = 2^(-5);
% %     filterParGuesses = logspace(0,-8,9);
% 
% %     mapParGuesses = linspace(0.1 , 10 , 10);
% %     mapParGuesses = 0.561;
%     mapParGuesses = 4;
%     
%     maxRankVec = 100:100:1000;
%     
%     rfTrainTime = [];
%     rfTestPerformance = [];
%     
%     for maxRank = maxRankVec
%     
%         alg = incrementalrfrls(map , maxRank , ...
%                                 'numRFParGuesses' , numRFParGuesses ,...
%                                 'mapParGuesses' , mapParGuesses ,  ...
%                                 'filterParGuesses', filterParGuesses , ...
%                                 'verbose' , 0 , ...
%                                 'storeFullTrainPerf' , storeFullTrainPerf , ...
%                                 'storeFullValPerf' , storeFullValPerf , ...
%                                 'storeFullTestPerf' , storeFullTestPerf, ...
%                                 'storeFullTrainTime' , storeFullTrainTime);
% 
%         expRFInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
%         expRFInc.run();
% 
%         rfTrainTime = [rfTrainTime , expRFInc.algo.trainTime];
%         rfTestPerformance = [rfTestPerformance , expRFInc.algo.testPerformance];
% 
%     end
% 
%     RFInc_cumulative_training_time(k) = expRFInc.time.train;
%     RFInc_cumulative_testing_time(k) = expRFInc.time.test;
%     RFInc_cumulative_test_perf(k) = expRFInc.result.perf;

%     % incrementalrfrls_plots

    %% Fastfood Gaussian Kernel approx RLS

%     map = @fastfoodGaussian;
%     fil = @tikhonov;
% 
%     filterParGuesses = logspace(-5,0,6);
% %     filterParGuesses = expKRLS.algo.filterParStar;
%     
%     alg =  ffrls(map , fil , 500 , 'numMapParGuesses' , 1 ,  ...
%                             'numMapParRangeSamples' , 1000 ,  ...
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf);
%                         
%     expFFRLS = experiment(alg , ds , 1 , true , saveResult , 'nm' , resdir , 0);
%     expFFRLS.run();
%     expFFRLS.result
% 
%     FFRLS_cumulative_training_time(k) = expFFRLS.time.train;
%     FFRLS_cumulative_testing_time(k) = expFFRLS.time.test;
%     FFRLS_cumulative_test_perf(k) = expFFRLS.result.perf;
%     
    
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
%     boxplot(nysTestPerformance , median(nysTrainTime) ,  'plotstyle' , 'compact' , 'positions' , median(nysTrainTime))
    boxplot(nysTestPerformance)
    ylabel('Test RMSE')
%     xlabel('Training time (s)')
%     legend('Inc Nys','RKS')
    
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