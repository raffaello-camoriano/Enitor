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
storeFullValPerf = 0;
storeFullTestPerf = 1;
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

ntrds = 4096; % max 4096
nteds = 4096; % max 4096
d = 32;
linearity = 'n';
noise = 'h';
dsExtraProps = {d,noise,linearity};
shuffleTraining = 1;
shuffleTest = 1;
shuffleAll = 1;
perfEvalStep = 20;

for k = 1:numRep

    display([ 'Repetition #', num2str(k)])
     
    % Load dataset
    ds = pumadyn(ntrds, nteds, [] , shuffleTraining, shuffleTest, shuffleAll, dsExtraProps);
    

    %% Incremental Nystrom KRLS

    map = @nystromUniformIncremental;

    numNysParGuesses = 2000;
%     filterParGuesses = logspace(0,-9,10);
    filterParGuesses = logspace(6,-13,10);
    mapParGuesses = 2.6667;
%     mapParGuesses = linspace(2,4,10);

    alg = incrementalNkrls2(map , 2000 , ...
                            'minRank' , 1 , ...
                            'numNysParGuesses' , numNysParGuesses ,...
                            'mapParGuesses' , mapParGuesses ,  ...
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf, ...
                            'perfEvalStep' , perfEvalStep);
                        
%     alg = incrementalNkrls(map , 1000 , ...
%                             'minRank' , 1 , ...
%                             'numNysParGuesses' , numNysParGuesses ,...
%                             'numMapParGuesses' , 10 ,  ...
%                             'numMapParRangeSamples' , 1000 ,  ...
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
    
    nysValPerformance = [nysValPerformance ; expNysInc.algo.valPerformance'];
    nysTestPerformance = [nysTestPerformance ; expNysInc.algo.testPerformance'];
    
    
%     expNysInc.result
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
%     filterParGuesses = 1e-7;
%     mapParGuesses = 2.6667;
% 
%     alg = nrls(map , filter , 1000 , ...
%                             'mapParGuesses' , mapParGuesses ,  ... 
%                             'filterParGuesses', filterParGuesses , ...
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
%     NysBat_cumulative_training_time(k) = expNysBat.time.train;
%     NysBat_cumulative_testing_time(k) = expNysBat.time.test;
%     NysBat_cumulative_test_perf(k) = expNysBat.result.perf;
% 
%     % incrementalnkrls_plots
% 
% 
%     
end

% save('wspace_plot1_adult_laptop.mat' , '-v7.3');


%% Plot 1 nips15

%% Incremental Nystrom performance (only val)

%%%% OLD
% figure
% % imagesc()
% pcolor(expNysInc.algo.filterParGuesses,expNysInc.algo.nyMapper.rng(1,:),expNysInc.algo.valPerformance)
% title({'Incremental Nystrom performance';'Validation Set'})
% ylabel('m')
% xlabel('\lambda')
% set(gca,'XScale','log')
% h = colorbar;
% h.Label.String = 'RMSE';

%%
% figure
% pcolor(expNysInc.algo.nyMapper.rng(1,:),expNysInc.algo.filterParGuesses,expNysInc.algo.valPerformance')
% % title({'Incremental Nystrom performance';'Validation Set'})
% xlabel('m')
% ylabel('\lambda')
% set(gca,'YScale','log')
% h = colorbar;
% h.Label.String = 'RMSE';

%% Interpolation

m = expNysInc.algo.nyMapper.rng(1,perfEvalStep:perfEvalStep:end);
l = expNysInc.algo.filterParGuesses;

Xdata = repmat(m, size(nysTestPerformance,1) , 1);
Ydata = repmat(l', 1, size(nysTestPerformance, 2));
Zdata = mean(nysTestPerformance,3);

% Generate query points
Xq = m;
Yq = 1e-9 ./ m;


Vq = interp2(Xdata,Ydata,Zdata,Xq,Yq,'cubic');




%%

% INCREMENTAL

figure
% pcolor(m,l,mean(nysTestPerformance,3))
contourf(m,l,mean(nysTestPerformance,3))
% Create ylabel
ylabel('\lambda','FontSize',36,'Rotation',0);
% Create xlabel
xlabel('m','FontSize',36);
set(gca,'FontSize',14);
set(gca,'YScale','log')
h = colorbar('FontSize',14);
h.Label.String = 'RMSE';
h.Label.FontSize = 20;



% BATCH

% figure
% m = cell2mat(expNysBat.algo.nyMapper.rng(1,:));
% l = expNysBat.algo.filterParGuesses;
% pcolor(m(1,:),l,mean(valPerf,3))
% % Create ylabel
% ylabel('\lambda','FontSize',36,'Rotation',0);
% % Create xlabel
% xlabel('m','FontSize',36);
% set(gca,'FontSize',14);
% set(gca,'YScale','log')
% h = colorbar('FontSize',14);
% h.Label.String = 'RMSE';
% h.Label.FontSize = 20;




%%
% figure
% hold on
% title({'Incremental Nystrom performance';'Validation Set'})
% colormap jet
% cc=jet(size(expNysInc.algo.nyMapper.rng(1,:),2));    
% for i = 1:size(expNysInc.algo.nyMapper.rng(1,:),2)
%     plot(expNysInc.algo.filterParGuesses,expNysInc.algo.valPerformance(i,:),'color',cc(i,:))
% end
% ylabel('RMSE')
% xlabel('\lambda')
% set(gca,'XScale','log')
% h = colorbar('Ticks' , 0:1/(numel(expNysInc.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysInc.algo.nyMapper.rng(1,:) );
% h.Label.String = 'm';
