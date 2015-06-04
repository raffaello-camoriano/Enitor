setenv('LC_ALL','C');
% addpath(genpath('.'));
% addpath(genpath('/home/iit.local/rcamoriano/repos/Enitor'));
 
clearAllButBP;
% close all;

% Set experimental results relative directory name
resdir = '';
mkdir(resdir);

%% Initialization

storeFullTrainPerf = 0;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 0;
saveResult = 0;

% Load dataset

numRep = 20;
trainPerf =  zeros(20,20,numRep );
testPerf =  zeros(20,20,numRep );
valPerf =  zeros(20,20,numRep );

for i = 1:numRep
    
    ds = breastCancer(400,169,'zeroOne');    

    for sigma = 0.9

        %% Batch Nystrom KRLS

        map = @nystromUniform;
        filter = @tikhonov;
        numNysParGuesses = 20;
        mapParGuesses = sigma;
%         filterParGuesses = 1e-5;
        filterParGuesses = logspace(-3,-12,20);


        alg = nrls(map , filter , 300 , ...
                                'minRank' , 5 , ...
                                'numNysParGuesses', numNysParGuesses , ...
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


    end
    
    if storeFullTrainPerf == 1
        trainPerf(:,:,i) = expNysBat.algo.trainPerformance';
    end
    if storeFullValPerf == 1
        valPerf(:,:,i) = expNysBat.algo.valPerformance';
    end
    if storeFullTestPerf == 1
        testPerf(:,:,i) = expNysBat.algo.testPerformance';
    end

end

mtrainperf = mean(trainPerf,3);
mvalperf = mean(valPerf,3);
mtestperf = mean(testPerf,3);


%%

figure
m = cell2mat(expNysBat.algo.nyMapper.rng(1,:));
l = expNysBat.algo.filterParGuesses;
pcolor(m(1,:),l,mean(valPerf,3))
% Create ylabel
ylabel('\lambda','FontSize',36,'Rotation',0);
% Create xlabel
xlabel('m','FontSize',36);
set(gca,'FontSize',14);
set(gca,'YScale','log')
h = colorbar('southoutside','FontSize',14);
h.Label.String = 'Classification Error';
h.Label.FontSize = 20;


%%
% 
% figure
% m = cell2mat(expNysBat.algo.nyMapper.rng(1,:));
% pcolor(m(1,:),expNysBat.algo.filterParGuesses,mean(valPerf,3))
% xlabel('m')
% ylabel('\lambda')
% set(gca,'YScale','log')
% h = colorbar;
% h.Label.String = 'RMSE';
% 
% 


%%
% figure
% % area(perfVec)
% boxplot(perfVec , 'plotstyle' , 'compact')