setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;
% close all;

% Set experimental results relative directory name
resdir = '';
mkdir(resdir);

%% Initialization

numRep =  1;
storeFullTrainPerf = 0;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 0;
saveResult = 0;

%% Storage vars init

for k = 1:numRep

    % Load dataset
    ntr = 1000;
    nte = 100;
    d = 5;
    noisesd = 1;
    ds = IIRproj_datagen(ntr , nte, d , noisesd);

%     ntr  =6554;
%     nte = 1638;
%     ds = cpuSmall(ntr  ,nte);

%     ntr  =500;
%     nte = 169;
%     ds = breastCancer(ntr  ,nte , 'zeroOne');

    % ds = Adult(7000,16282,'plusMinusOne');
%     ds = Adult(ntr,16282,'plusMinusOne');
    
    numEpochs = 400;

    
    %% IIR Experiment setup

    map = @gaussianKernel;
    fil = @gdesc2_square_loss;
    maxiterIIR = numEpochs*round(ntr*0.8);

    alg = gdesc( map , fil , ...
                    'filterParGuesses' , 1:maxiterIIR , ...
                    'verbose' , 0 , ...
                    'storeFullTrainPerf' , storeFullTrainPerf , ...
                    'storeFullValPerf' , storeFullValPerf , ...
                    'storeFullTestPerf' , storeFullTestPerf);

    expIIR = experiment(alg , ds , 1 , true , saveResult , '' , resdir);

    expIIR.run();
    expIIR.result

        %% Experiment 1 setup, Landweber, Gaussian kernel

%     map = @linearKernel;
%     fil = @gdesc_square_loss;
%     maxiter = 150;
% 
% 
%     alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , 1:maxiter   , 'verbose' , 0 , ...
%                             'storeFullTrainPerf' , storeFullTrainPerf , 'storeFullValPerf' , storeFullValPerf , 'storeFullTestPerf' , storeFullTestPerf);
% 
%     expLandweber = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
% 
%     expLandweber.run();
%     expLandweber.result
% 
%     Landweber_cumulative_training_time(k) = expLandweber.time.train;
%     Landweber_cumulative_testing_time(k) = expLandweber.time.test;
%     Landweber_cumulative_test_perf(k) = expLandweber.result.perf;
%     zoomws.eps
%     
%     % landweber_plots
% 
%     % plot_1_padova

    
end

%% IIR graph
fig=figure;
hax=axes;
hold on
% plot(expIIR.algo.valPerformance) 
% semilogy(expIIR.algo.valPerformance) 
loglog(1:maxiterIIR,expIIR.algo.valPerformance) 
% SP = 0;
% while (SP < maxiterIIR)
%     line([SP SP],get(hax,'YLim'),'Color',[1 0 0])
%     SP=SP+round(ntr*0.8); %your point goes here
% end

%%
% 
% plots
% 
% %% Save figures
% figsdir = resdir;
% % % mkdir(figsdir);
% saveAllFigs