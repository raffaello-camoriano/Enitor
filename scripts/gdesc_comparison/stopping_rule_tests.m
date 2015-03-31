setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = 'scripts/gdesc_comparison/padova_plots/';
mkdir(resdir);

%% Initialization

numRep =  1;
storeFullTrainPerf = 0;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 0;

%% Storage vars init

% Training time
KRLS_cumulative_training_time= zeros(numRep,1);
DACKRLS_cumulative_training_time= zeros(numRep,1);
Landweber_cumulative_training_time= zeros(numRep,1);
NuMethod_cumulative_training_time= zeros(numRep,1);
NysInc_cumulative_training_time= zeros(numRep,1);

% Testing time
KRLS_cumulative_testing_time= zeros(numRep,1);
DACKRLS_cumulative_testing_time= zeros(numRep,1);
Landweber_cumulative_testing_time= zeros(numRep,1);
NuMethod_cumulative_testing_time= zeros(numRep,1);
NysInc_cumulative_testing_time= zeros(numRep,1);


% Test performance
KRLS_cumulative_test_perf = zeros(numRep,1);
DACKRLS_cumulative_test_perf= zeros(numRep,1);
Landweber_cumulative_test_perf= zeros(numRep,1);
NuMethod_cumulative_test_perf= zeros(numRep,1);
NysInc_cumulative_test_perf = zeros(numRep,1);

for k = 1:numRep

    % Load dataset
    % ds = Adult(7000,16282,'plusMinusOne');
    ds = Adult(1000,16282,'plusMinusOne');
    
    %% Experiment 1 setup, Landweber, Gaussian kernel

    map = @gaussianKernel;
    fil = @gdesc_square_loss;
    maxiter = 7000;
%     stoppingRule = sharpStop(2);
%     stoppingRule = tolerantStop(2,0.005);
    stoppingRule = horizonSharpStop(2,maxiter/20);


    alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , 1:maxiter   , 'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf , ...
                            'stoppingRule' , stoppingRule);

%     alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , 1:maxiter   , 'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf);

    expLandweber = experiment(alg , ds , 1 , true , true , '' , resdir);

    expLandweber.run();
    expLandweber.result

    Landweber_cumulative_training_time(k) = expLandweber.time.train;
    Landweber_cumulative_testing_time(k) = expLandweber.time.test;
    Landweber_cumulative_test_perf(k) = expLandweber.result.perf;
    
    
    % landweber_plots

    % plot_1_padova

    %% Experiment 2 setup, nu method, Gaussian kernel

    map = @gaussianKernel;
    fil = @numethod_square_loss;
    maxiter = 100;
    stoppingRule = horizonSharpStop(2,maxiter/20);

%     alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , ...
%                             'filterParGuesses' , 1:maxiter   ,...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf ,...
%                             'storeFullValPerf' , storeFullValPerf ,...
%                             'storeFullTestPerf' , storeFullTestPerf);
                        
    alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , ...
                            'filterParGuesses' , 1:maxiter   ,...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf ,...
                            'storeFullValPerf' , storeFullValPerf ,...
                            'storeFullTestPerf' , storeFullTestPerf, ...
                            'stoppingRule' , stoppingRule);

    expNuMethod = experiment(alg , ds , 1 , true , true , '' , resdir);

    expNuMethod.run();
    expNuMethod.result


    NuMethod_cumulative_training_time(k) = expNuMethod.time.train;
    NuMethod_cumulative_testing_time(k) = expNuMethod.time.test;
    NuMethod_cumulative_test_perf(k) = expNuMethod.result.perf;
    
    % numethod_plots


    %% Exact KRLS

    map = @gaussianKernel;
    fil = @tikhonov;

    filterParGuesses = logspace(-5,0,30);

    alg = krls( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , filterParGuesses , 'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf , 'storeFullTestPerf' , storeFullTestPerf);

    % alg = krls( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , logspace(-5,0,12) , 'verbose' , 0 , ...
    %                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);

    % alg = krls( map , fil , 'numMapParGuesses' , 10 , 'numFilterParGuesses' , 10 , 'verbose' , 1 , ...
    %                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);

    % alg = krls( map , fil , 'mapParGuesses' , linspace(1,5,10) , 'filterParGuesses' , logspace(-5,1,10) , 'verbose' , 1 , ...
    %                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);

    expKRLS = experiment(alg , ds , 1 , true , true , 'nh' , resdir);

    expKRLS.run();
    expKRLS.result

    KRLS_cumulative_training_time(k) = expKRLS.time.train;
    KRLS_cumulative_testing_time(k) = expKRLS.time.test;
    KRLS_cumulative_test_perf(k) = expKRLS.result.perf;

    % krls_plots

    %% Divide & Conquer KRLS

    % Algorithm init
    %     figure
    %     title('Batch Nystrom performance')
    %     hold on    
    %     plot(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.trainPerformance);
    %     plot(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.valPerformance);
    % %     plot(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.testPerformance);
    %     hold off
    %     ylabel('\sigma','fontsize',16)
    %     xlabel('\lambda','fontsize',16)
    %     legend('Training','Validation');    
    %     set(gca,'XScale','log')
    map = @gaussianKernel;  
    fil = @tikhonov;
    % mGuesses = 2:10;
    mGuesses = 2;
    mapParGuesses = expKRLS.algo.mapParStar;
    mapParStarIdx = find(expKRLS.algo.mapParGuesses==mapParGuesses);
    filterParGuesses = expKRLS.algo.filterParGuessesStorage(mapParStarIdx,:);


    alg = dackrls(map , fil , mGuesses , 'mapParGuesses' , mapParGuesses , 'filterParGuesses' , filterParGuesses ,...
        'verbose' , verbose , 'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf ,...
        'storeFullTestPerf' , storeFullTestPerf);


    % alg = dackrls(map , fil , mGuesses , 'mapParGuesses' , mapParGuesses , 'filterParGuesses' , filterParGuesses ,...
    %     'verbose' , verbose , 'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf ,...
    %     'storeFullTestPerf' , storeFullTestPerf);
    % Exp init
    expDACKRLS = experiment(alg , ds , 1 , true , true , '' , resdir);

    expDACKRLS.run();
    expDACKRLS.result

    DACKRLS_cumulative_training_time(k) = expDACKRLS.time.train;
    DACKRLS_cumulative_testing_time(k) = expDACKRLS.time.test;
    DACKRLS_cumulative_test_perf(k) = expDACKRLS.result.perf;

    % dackrls_plots


    %% Incremental Nystrom KRLS

    map = @nystromUniformIncremental;

    numNysParGuesses = 20;
    stoppingRule = horizonSharpStop(2,maxiter/20);

%     alg = incrementalNkrls(map , 200 , 'numNysParGuesses' , numNysParGuesses ,...
%                             'mapParGuesses' , mapParGuesses ,  ...
%                             'filterParGuesses', filterParGuesses , 'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf);

    alg = incrementalNkrls(map , 200 , 'numNysParGuesses' , numNysParGuesses ,...
                            'mapParGuesses' , mapParGuesses ,  ...
                            'filterParGuesses', filterParGuesses , 'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf, ...
                            'stoppingRule' , stoppingRule);

    expNysInc = experiment(alg , ds , 1 , true , true , 'nm' , resdir , 0);
    expNysInc.run();
    expNysInc.result

    NysInc_cumulative_training_time(k) = expNysInc.time.train;
    NysInc_cumulative_testing_time(k) = expNysInc.time.test;
    NysInc_cumulative_test_perf(k) = expNysInc.result.perf;

    % incrementalnkrls_plots

    %% Random Features KRLS


end


%% Plot timing

figure
trainingTimes = [ expKRLS.result.time.train , expDACKRLS.result.time.train , expNysInc.result.time.train , expLandweber.result.time.train , expNuMethod.result.time.train ];
bar(trainingTimes)
set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'incNKRLS', 'Landweber' , '\nu method'})
title('Training & Model Selection Times')
ylabel('Time (s)')

figure
trainingTimes = [ expKRLS.result.time.test , expDACKRLS.result.time.test , expNysInc.result.time.test , expLandweber.result.time.test , expNuMethod.result.time.test];
bar(trainingTimes)
set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'incNKRLS', 'Landweber' , '\nu method'})
title('Testing Times')
ylabel('Time (s)')

%% Plot best test performances

figure
testPerf = [ expKRLS.result.perf , expDACKRLS.result.perf , expNysInc.result.perf , expLandweber.result.perf , expNuMethod.result.perf];
bar(testPerf)
set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'incNKRLS', 'Landweber' , '\nu method'})
title('Best test performance')
ylabel('Relative Error')

%%
% 
plot_2_padova
% 
% %% Save figures
% figsdir = resdir;
% % mkdir(figsdir);
% saveAllFigs