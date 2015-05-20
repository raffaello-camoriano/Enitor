setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = '';
% mkdir(resdir);

%% Initialization

numRep = 10;
storeFullTrainPerf = 0;
storeFullValPerf = 1;
storeFullTestPerf = 1;
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
    ds = Covertype(522910,58102,'plusMinusOne');
    
    %% Experiment 1 setup, Landweber, Gaussian kernel
% 
%     map = @gaussianKernel;
%     fil = @gdesc_square_loss;
%     maxiter = 7000;
% 
% 
%     alg = kgdesc(map, fil , 'numMapParGuesses' , 1 , ...
%                             'filterParGuesses' , 1:maxiter , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf);
% 
%     expLandweber = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
% 
%     expLandweber.run();
%     expLandweber.result
% 
%     Landweber_cumulative_training_time(k) = expLandweber.time.train;
%     Landweber_cumulative_testing_time(k) = expLandweber.time.test;
%     Landweber_cumulative_test_perf(k) = expLandweber.result.perf;
%     
%     
%     % landweber_plots
% 
%     % plot_1_padova

    %% Experiment 2 setup, nu method, Gaussian kernel

%     map = @gaussianKernel;
%     fil = @numethod_square_loss;
%     maxiter = 3000;
% 
%     alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , ...
%                               'filterParGuesses' , 1:maxiter , ...
%                               'verbose' , 0 , ...
%                               'storeFullTrainPerf' , storeFullTrainPerf , ...
%                               'storeFullValPerf' , storeFullValPerf , ...
%                               'storeFullTestPerf' , storeFullTestPerf);
% 
%     expNuMethod = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
% 
%     expNuMethod.run();
%     expNuMethod.result
% 
% 
%     NuMethod_cumulative_training_time(k) = expNuMethod.time.train;
%     NuMethod_cumulative_testing_time(k) = expNuMethod.time.test;
%     NuMethod_cumulative_test_perf(k) = expNuMethod.result.perf;
%     
%     % numethod_plots
    
    %% Experiment 3 setup, subgradient descent, hinge loss, Gaussian kernel

%     map = @gaussianKernel;
%     fil = @gdesc_kernel_hinge_loss;
%     maxiter = 7000;
% 
% 
%     alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , ...
%         'filterParGuesses' , 1:maxiter   , ...
%         'verbose' , 0 , ...
%         'storeFullTrainPerf' , storeFullTrainPerf , ...
%         'storeFullValPerf' , storeFullValPerf , ...
%         'storeFullTestPerf' , storeFullTestPerf);
% 
%     expgdesc_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
% 
%     expgdesc_kernel_hinge_loss.run();
%     expgdesc_kernel_hinge_loss.result
% 
%     gdesc_kernel_hinge_loss_cumulative_training_time(k) = expgdesc_kernel_hinge_loss.time.train;
%     gdesc_kernel_hinge_loss_cumulative_testing_time(k) = expgdesc_kernel_hinge_loss.time.test;
%     gdesc_kernel_hinge_loss_cumulative_test_perf(k) = expgdesc_kernel_hinge_loss.result.perf;
%     
    
    %% Exact KRLS
% 
%     map = @gaussianKernel;
%     fil = @tikhonov;
% 
%     filterParGuesses = logspace(0,-8,9);
% 
%     alg = krls( map , fil , 'numMapParGuesses' , 1 , ...
%                 'filterParGuesses' , filterParGuesses , ...
%                 'verbose' , 0 , ...
%                 'storeFullTrainPerf' , storeFullTrainPerf , ...
%                 'storeFullValPerf' , storeFullValPerf , ...
%                 'storeFullTestPerf' , storeFullTestPerf);
% 
%     expKRLS = experiment(alg , ds , 1 , true , saveResult , 'nh' , resdir);
% 
%     expKRLS.run();
%     expKRLS.result
% 
%     KRLS_cumulative_training_time(k) = expKRLS.time.train;
%     KRLS_cumulative_testing_time(k) = expKRLS.time.test;
%     KRLS_cumulative_test_perf(k) = expKRLS.result.perf;

    % krls_plots

    %% Divide & Conquer KRLS
% 
%     % Algorithm init
%     %     figure
%     %     title('Batch Nystrom performance')
%     %     hold on    
%     %     plot(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.trainPerformance);
%     %     plot(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.valPerformance);
%     % %     plot(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.testPerformance);
%     %     hold off
%     %     ylabel('\sigma','fontsize',16)
%     %     xlabel('\lambda','fontsize',16)
%     %     legend('Training','Validation');    
%     %     set(gca,'XScale','log')
%     map = @gaussianKernel;  
%     fil = @tikhonov;
%     % mGuesses = 2:10;
%     mGuesses = 2;
%     mapParGuesses = expKRLS.algo.mapParStar;
% %     mapParStarIdx = find(expKRLS.algo.mapParGuesses==mapParGuesses);
% %     filterParGuesses = expKRLS.algo.filterParGuessesStorage(mapParStarIdx,:);
%     filterParGuesses = logspace(0,-5,30);
% 
%     alg = dackrls(map , fil , mGuesses , 'mapParGuesses' , mapParGuesses , 'filterParGuesses' , filterParGuesses ,...
%         'verbose' , verbose , 'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf ,...
%         'storeFullTestPerf' , storeFullTestPerf);
% 
% 
%     % alg = dackrls(map , fil , mGuesses , 'mapParGuesses' , mapParGuesses , 'filterParGuesses' , filterParGuesses ,...
%     %     'verbose' , verbose , 'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf ,...
%     %     'storeFullTestPerf' , storeFullTestPerf);
%     % Exp init
%     
%     expDACKRLS = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
% 
%     expDACKRLS.run();
%     expDACKRLS.result
% 
%     DACKRLS_cumulative_training_time(k) = expDACKRLS.time.train;
%     DACKRLS_cumulative_testing_time(k) = expDACKRLS.time.test;
%     DACKRLS_cumulative_test_perf(k) = expDACKRLS.result.perf;

    % dackrls_plots


    %% Incremental Nystrom KRLS

    map = @nystromUniformIncremental;

    numNysParGuesses = 10;
    filterParGuesses = 1e-9;
    mapParGuesses = 1.12222;
    
    alg = incrementalNkrls(map , 1000 , ...
                            'minRank' , 100 , ...
                            'numNysParGuesses' , numNysParGuesses ,...
                            'mapParGuesses' , mapParGuesses ,  ... 
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf , ...
                            'storeFullTrainTime' , storeFullTrainTime);

%     alg = incrementalNkrls(map , 1000 , 'numNysParGuesses' , numNysParGuesses ,...
%                             'numMapParGuesses' , 10 ,  ...
%                             'numMapParRangeSamples' , 5000 ,  ...
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf);

%     alg.mapParStar = [400 , 5];
%     alg.filterParStar = 1e-8;
%     alg.justTrain(ds.X(ds.trainIdx,:) , ds.Y(ds.trainIdx));
% 
%     YtePred = alg.test(ds.X(ds.testIdx,:));   
%       
%     perf = abs(ds.performanceMeasure( ds.Y(ds.testIdx,:) , YtePred , ds.testIdx));

    expNysInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
    expNysInc.run();
%     expNysInc.result

    nysTrainTime = [nysTrainTime ; expNysInc.algo.trainTime'];
    nysTestPerformance = [nysTestPerformance ; expNysInc.algo.testPerformance'];

    NysInc_cumulative_training_time(k) = expNysInc.time.train;
    NysInc_cumulative_testing_time(k) = expNysInc.time.test;
    NysInc_cumulative_test_perf(k) = expNysInc.result.perf;

    % incrementalnkrls_plots


     %% Batch Random Features RLS
% 
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
% 
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
    boxplot(nysTestPerformance , median(nysTrainTime) ,  'plotstyle' , 'compact' , 'positions' , median(nysTrainTime))
    ylabel('Test RMSE')
    xlabel('Training time (s)')
    legend('Inc Nys','RKS')
    
    
end



%%
% % 
% % plots
% % 
% % %% Save figures
% figsdir = resdir;
% % % mkdir(figsdir);
% saveAllFigs