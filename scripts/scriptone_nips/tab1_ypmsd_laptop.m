setenv('LC_ALL','C');
addpath(genpath('.'));
% addpath(genpath('/home/iit.local/rcamoriano/repos/Enitor'));
 
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
nysValPerformance = [];

for k = 1:numRep

    display([ 'Repetition #', num2str(k)])
     
    % Load dataset
    ds = YearPredictionMSD(463715,51630);


    %% Incremental Nystrom KRLS

    map = @nystromUniformIncremental;

    numNysParGuesses = 2;
%     numNysParGuesses = 1;
%     filterParGuesses = expKRLS.algo.filterParStar;
%     filterParGuesses = logspace(0,-7,8);
    filterParGuesses = 1e-8;
%     mapParGuesses = linspace(1 , 2, 10);
%     mapParGuesses = [4 6 7 8 10];
    mapParGuesses = 1.8889 ;

    alg = incrementalNkrls(map , 2048 , ...
                            'minRank' , 10 , ...
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
%     
%     if numRep > 1
%         nysValPerformance = [nysValPerformance ; expNysInc.algo.valPerformance'];
%     end
    
    
%     expNysInc.result
% 
%     NysInc_cumulative_training_time(k) = expNysInc.time.train;
%     NysInc_cumulative_testing_time(k) = expNysInc.time.test;
%     NysInc_cumulative_test_perf(k) = expNysInc.result.perf;

    % incrementalnkrls_plots


    %% Incremental Random Features RLS
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

save('wspace.mat' , '-v7.3');




%% Plot 1 nips15

if numRep == 1
   
    % Plot perf

    for startIdxM = 1:numNysParGuesses:size(expNysInc.algo.mapParGuesses,2)
        
        endIdxM = startIdxM + numNysParGuesses - 1;
        sigma = expNysInc.algo.mapParGuesses(2,startIdxM);
        
        figure
        hold on
        plot( expNysInc.algo.mapParGuesses(1,startIdxM:endIdxM), expNysInc.algo.valPerformance(startIdxM:endIdxM) , 'Marker' , 'diamond')
        title(['Validation Error for \sigma = ' , num2str(sigma)])
        ylabel('Validation error')
        xlabel('m')
    end
end

if numRep > 1
    
    % Plot perf

    for startIdxM = 1:numNysParGuesses:size(expNysInc.algo.mapParGuesses,2)
        
        endIdxM = startIdxM + numNysParGuesses - 1;
        sigma = expNysInc.algo.mapParGuesses(2,startIdxM);
        
        figure
        hold on
        boxplot(nysValPerformance(:,startIdxM:endIdxM) , expNysInc.algo.mapParGuesses(1,startIdxM:endIdxM) ,  'plotstyle' , 'compact')
        title(['Validation Error for \sigma = ' , num2str(sigma)])
        ylabel('Validation error')
        xlabel('m')
    end
end


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
