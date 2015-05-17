setenv('LC_ALL','C');
% addpath(genpath('.'));
addpath('/home/rcamoriano/repos/Enitor');
 
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = 'scripts/incrementalrftests/plots/';
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

for k = 1:numRep

    display([ 'Repetition #', num2str(k)])
     
    % Load dataset
   % ds = Adult(7000,16282,'plusMinusOne');
%     ds = Adult(2000,16282,'plusMinusOne');
%     ds = Adult(2500,16282,'plusMinusOne');
    ds = Cifar10(5000,1000,'plusMinusOne',0:9);
%     ds = Covertype(522910,58102,'plusOneMinusBalanced');
%     ds = YearPredictionMSD(463715,51630);
    
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
%     filterParGuesses = expKRLS.algo.filterParStar;
%     filterParGuesses = logspace(0,-9,10);
    filterParGuesses = 1e-10;
%     mapParGuesses = linspace(0.1 , 1 , 10);
%     mapParGuesses = linspace(0.3 , 0.1 , 10);

%     alg = incrementalNkrls(map , 1000 , ...
%                             'numNysParGuesses' , numNysParGuesses ,...
%                             'mapParGuesses' , mapParGuesses ,  ...
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf);
                        
    alg = incrementalNkrls(map , 1000 , ...
                            'numNysParGuesses' , numNysParGuesses ,...
                            'numMapParGuesses' , 10 ,  ...
                            'numMapParRangeSamples' , 10000 ,  ...
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf);

    alg.mapParStar = [26757 , 7];
    alg.filterParStar = 1e-8;
    alg.justTrain(ds.X(ds.trainIdx,:) , ds.Y(ds.trainIdx));

    YtePred = obj.algo.test(Xte);   
      
    perf = abs(ds.performanceMeasure( Yte , YtePred , ds.testIdx));

%     expNysInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
%     expNysInc.run();
%     expNysInc.result
% 
%     NysInc_cumulative_training_time(k) = expNysInc.time.train;
%     NysInc_cumulative_testing_time(k) = expNysInc.time.test;
%     NysInc_cumulative_test_perf(k) = expNysInc.result.perf;

    % incrementalnkrls_plots


%     %% Incremental Random Features RLS
% 
%     map = @randomFeaturesGaussianIncremental;
% 
%     numRFParGuesses = 10;
% %     filterParGuesses = expKRLS.algo.filterParStar;
%     filterParGuesses = logspace(0,-9,10);
% %     filterParGuesses = 1e-8;
%     
%     alg = incrementalrfrls(map , 15000 , 'numRFParGuesses' , numRFParGuesses ,...
%                             'numMapParGuesses' , 40 ,  ...
%                             'numMapParRangeSamples' , 10000 ,  ...
%                             'filterParGuesses', filterParGuesses , ...
%                             'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , ...
%                             'storeFullValPerf' , storeFullValPerf , ...
%                             'storeFullTestPerf' , storeFullTestPerf);
% 
%     expRFInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
%     expRFInc.run();
%     expRFInc.result
% 
%     RFInc_cumulative_training_time(k) = expRFInc.time.train;
%     RFInc_cumulative_testing_time(k) = expRFInc.time.test;
%     RFInc_cumulative_test_perf(k) = expRFInc.result.perf;

    % incrementalrfrls_plots

    %% Fastfood Gaussian Kernel approx RLS

%     map = @fastfoodGaussian;
%     fil = @tikhonov;
% 
%     numRFParGuesses = 20;
% %     filterParGuesses = logspace(-5,0,10);
% %     filterParGuesses = expKRLS.algo.filterParStar;
%     
%     alg =  ffrls(map , 200 , fil,  1CEST , 7, 500);
%                         
%     expFFRLS = experiment(alg , ds , 1 , true , saveResult , 'nm' , resdir , 0);
%     expFFRLS.run();
%     expFFRLS.result
% 
%     FFRLS_cumulative_training_time(k) = expFFRLS.time.train;
%     FFRLS_cumulative_testing_time(k) = expFFRLS.time.test;
%     FFRLS_cumulative_test_perf(k) = expFFRLS.result.perf;
    
    
end

%% Plots
% 
% if numRep == 1
%     % Plot timing
%     % figure
%     % trainingTimes = [ expKRLS.result.time.train , expDACKRLS.result.time.train , expNysInc.result.time.train , expRFInc.result.time.train , expLandweber.result.time.train , expNuMethod.result.time.train expgdesc_kernel_hinge_loss.result.time.train , expFFRLS.result.time.train];
%     % bar(trainingTimes)
%     % set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'incNKRLS', 'incRFRLS', 'Landweber' , '\nu method' , 'Subgr. SVM' , 'Fastfood'})
%     % title('Training & Model Selection Times')
%     % ylabel('Time (s)')
%     % 
%     % figure
%     % trainingTimes = [ expKRLS.result.time.test , expDACKRLS.result.time.test , expNysInc.result.time.test , expRFInc.result.time.test , expLandweber.result.time.test , expNuMethod.result.time.test , expgdesc_kernel_hinge_loss.result.time.test , expFFRLS.result.time.test];
%     % bar(trainingTimes)
%     % set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'incNKRLS', 'incRFRLS', 'Landweber' , '\nu method', 'Subgr. SVM', 'Fastfood'})
%     % title('Testing Times')
%     % ylabel('Time (s)')
% 
%     figure
%     trainingTimes = [ expNysInc.result.time.train , expRFInc.result.time.train ];
%     bar(trainingTimes)
%     set(gca,'XTickLabel',{ 'incNKRLS', 'incRFRLS' })
%     title('Training & Model Selection Times')
%     ylabel('Time (s)')
% 
%     figure
%     trainingTimes = [  expNysInc.result.time.test , expRFInc.result.time.test];
%     bar(trainingTimes)
%     set(gca,'XTickLabel',{'incNKRLS', 'incRFRLS'})
%     title('Testing Times')
%     ylabel('Time (s)')
% 
%     %% Plot best test performances
% 
%     figure
%     testPerf = [  expNysInc.result.perf, expRFInc.result.perf   ];
%     bar(testPerf)
%     set(gca,'XTickLabel',{ 'incNKRLS', 'incRFRLS' })
%     title('Best test performance')
%     ylabel('Relative Error')
% 
% end
% 
% if numRep > 1
%     % Plot timing
%     figure
%     trainingTimesM = [ mean(expKRLS.result.time.train) , mean(expNysInc.result.time.train) , mean(expRFInc.result.time.train , expLandweber.result.time.train) , mean(expNuMethod.result.time.train) , mean(expgdesc_kernel_hinge_loss.result.time.train)];   
%     trainingTimesSD = [ std(expKRLS.result.time.train , 2) ,  std(expNysInc.result.time.train , 2) ,  std(expRFInc.result.time.train , 2) ,  std(expLandweber.result.time.train , 2) ,  std(expNuMethod.result.time.train , 2) ,  std(expgdesc_kernel_hinge_loss.result.time.train , 2)];   
%     x = 1:numel(trainingTimesM);
%     bar(trainingTimesM)
%     hold on
%     errorbar(x,trainingTimesM,trainingTimesSD,'rx')
%     set(gca,'XTickLabel',{'KRLS',  'incNKRLS', 'incRFRLS', 'Landweber' , '\nu method' , 'Subgr. SVM'})
%     title('Training & Model Selection Times')
%     ylabel('Time (s)')
% 
%     figure
%     testTimesM = [ mean(expKRLS.result.time.test) , mean(expNysInc.result.time.test) , mean(expRFInc.result.time.test) , mean(expLandweber.result.time.test) , mean(expNuMethod.result.time.test) , mean(expgdesc_kernel_hinge_loss.result.time.test)];
%     testTimesSD = [ std(expKRLS.result.time.test , 2)  , std(expNysInc.result.time.test , 2)  , std(expRFInc.result.time.test , 2)  , std(expLandweber.result.time.test , 2)  , std(expNuMethod.result.time.test , 2)  , std(expgdesc_kernel_hinge_loss.result.time.test , 2) ];
%     x = 1:numel(testTimesM);
%     bar(testTimesM)
%     hold on
%     errorbar(x,testTimesM,testTimesSD,'rx')
%     set(gca,'XTickLabel',{'KRLS', 'incNKRLS', 'incRFRLS', 'Landweber' , '\nu method', 'Subgr. SVM'})
%     title('Testing Times')
%     ylabel('Time (s)')
%     % Plot best test performances
% 
%     figure
%     testPerfM = [ mean(expKRLS.result.perf ),  mean(expNysInc.result.perf), mean(expRFInc.result.perf ), mean(expLandweber.result.perf ), mean(expNuMethod.result.perf) , mean(expgdesc_kernel_hinge_loss.result.perf) ];
%     testPerfSD = [ std(expKRLS.result.perf  , 2), std( expNysInc.result.perf , 2), std(expRFInc.result.perf  , 2), std(expLandweber.result.perf  , 2), std(expNuMethod.result.perf  , 2), std(expgdesc_kernel_hinge_loss.result.perf  , 2)];
%     x = 1:numel(testPerfM);
%     bar(testPerfM)
%     hold on
%     errorbar(x,testPerfSD,testPerfSD,'rx')
%     set(gca,'XTickLabel',{'KRLS', 'incNKRLS', 'incRFRLS' , 'Landweber' , '\nu method', 'Subgr. SVM'})
%     title('Best test performance')
%     ylabel('Relative Error')
% 
% end
% 
% %%
% % 
% % plots
% % 
% % %% Save figures
% figsdir = resdir;
% % % mkdir(figsdir);
% saveAllFigs