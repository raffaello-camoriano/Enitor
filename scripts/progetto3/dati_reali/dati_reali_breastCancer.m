setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;
% close all;

% Set experimental results relative directory name
resdir = '';
mkdir(resdir);

%% Initialization

numRep =  1;
storeFullTrainPerf = 1;
storeFullValPerf = 1;
storeFullTestPerf = 1;
verbose = 0;
saveResult = 0;
numEpochs = 10000;
% numEpochs = 10;

%% Storage vars init

for k = 1:numRep

    % Load dataset

    ntrval  =500;
    nte = 169;
    ds = breastCancer(ntrval  ,nte , 'zeroOne');
    dsRF = breastCancer(ntrval  ,nte , 'zeroOne');

    % Random Fourier features mapping

    ntr = numel(ds.trainIdx);
    maxRank = 500;
    sigma = 0.791;
    
    % Initialize Random Features Mapper
    argin = {};
    argin = [argin , 'maxRank' , maxRank];
    argin = [argin , 'mapParGuesses' , sigma];
    argin = [argin , 'verbose' , verbose];
    rfMapper = randomFeaturesGaussian(dsRF.X, dsRF.Y , ntr , argin{:} );
                            
    % Map samples with new hyperparameters
    rfMapper.next();
    rfMapper.compute();

    % Update dataset  object
    dsRF.X = rfMapper.Xrf;
    dsRF.d = maxRank;
    

    %% KRLS baseline Experiment setup

    display('KRLS baseline Experiment');
    map = @gaussianKernel;
    filter = @tikhonov;
    
    mapParGuesses = sigma;
    filterParGuesses = logspace(0,-9,10);

    alg = krls( map , filter , ...
                'mapParGuesses' , mapParGuesses , ...
                'filterParGuesses' , filterParGuesses , ...
                'verbose' , 0 , ...
                'storeFullTrainPerf' , storeFullTrainPerf , ...
                'storeFullValPerf' , storeFullValPerf , ...
                'storeFullTestPerf' , storeFullTestPerf);

    expKRLS = experiment(alg , ds , 1 , true , saveResult , '' , resdir);

    expKRLS.run();
    expKRLS.result
    
    
    %% IIR Experiment setup
    
    display('IIR Experiment');

    filter = @gdesc2_square_loss;
    maxiterIIR = numEpochs*round(ntr*0.8);

    alg = gdesc( filter , ...
                'filterParGuesses' , 1:maxiterIIR , ...
                'verbose' , 0 , ...
                'storeFullTrainPerf' , storeFullTrainPerf , ...
                'storeFullValPerf' , storeFullValPerf , ...
                'storeFullTestPerf' , storeFullTestPerf);

    expIIR = experiment(alg , dsRF , 1 , true , saveResult , '' , resdir);

    expIIR.run();
    expIIR.result

    %% Experiment 1 setup, Landweber, Gaussian kernel

    display('Gaussian kernel Landweber Experiment');
        
    map = @gaussianKernel;
    fil = @gdesc_kernel_square_loss;
%     maxiterLandw = numEpochs*round(ntr*0.8);
    maxiterLandw = numEpochs;

    alg = kgdesc( map , fil , ...
                'mapParGuesses' , sigma , ...
                'filterParGuesses' , 1:maxiterLandw , ...
                'verbose' , 0 , ...
                'storeFullTrainPerf' , storeFullTrainPerf , ...
                'storeFullValPerf' , storeFullValPerf , ...
                'storeFullTestPerf' , storeFullTestPerf);

    expLandweber = experiment(alg , ds , 1 , true , saveResult , '' , resdir);

    expLandweber.run();
    expLandweber.result

    Landweber_cumulative_training_time(k) = expLandweber.time.train;
    Landweber_cumulative_testing_time(k) = expLandweber.time.test;
    Landweber_cumulative_test_perf(k) = expLandweber.result.perf;
    
    % landweber_plots
% 
%     % plot_1_padova

    
end

%% KRLS graph
fig=figure;
hax=axes;
semilogx(1./filterParGuesses,expKRLS.algo.valPerformance) 
hold on
% semilogx(1./filterParGuesses,expKRLS.algo.valPerformance) 
title('KRLS validation error')
xlabel('1/\lambda')
ylabel('Error')
% plot(expIIR.algo.valPerformance) 
% semilogy(expIIR.algo.valPerformance) 
semilogx(1./filterParGuesses,expKRLS.algo.trainPerformance)
legend('Validation Error','Training Error')


% fig=figure;
% hax=axes;
% semilogx(1./filterParGuesses,expKRLS.algo.testPerformance) 
% title('KRLS test error')
% xlabel('1/\lambda')
% ylabel('Test error')

%% IIR graph
fig=figure;
hax=axes;
hold on
title('IIR validation error')
xlabel('Iterations')
ylabel('Error')
% plot(expIIR.algo.valPerformance) 
% semilogy(expIIR.algo.valPerformance) 
loglog(1:maxiterIIR,expIIR.algo.valPerformance) 
loglog(1:maxiterIIR,expIIR.algo.trainPerformance)
legend('Validation Error','Training Error')
% SP = 0;
% while (SP < maxiterIIR)
%     line([SP SP],get(hax,'YLim'),'Color',[1 0 0])
%     SP=SP+round(ntr*0.8); %your point goes here
% end


% fig=figure;
% hax=axes;
% plot(1:maxiterIIR,expIIR.algo.testPerformance) 
% title('IIR test error')
% xlabel('Iterations')
% ylabel('Test error')

%% Kernelized batch Landweber graph
fig=figure;
hax=axes;
hold on
title('Landweber validation error')
xlabel('Epochs')
ylabel('Error')
% plot(expIIR.algo.valPerformance) 
% semilogy(expIIR.algo.valPerformance) 
loglog(1:maxiterLandw,expLandweber.algo.valPerformance) 
loglog(1:maxiterLandw,expLandweber.algo.trainPerformance)
legend('Validation Error','Training Error')

% fig=figure;
% hax=axes;
% plot(1:maxiterLandw,expLandweber.algo.testPerformance) 
% title('Landweber test error')
% xlabel('Epochs')
% ylabel('Test error')

%%
% 
% plots
% 
% %% Save figures
% figsdir = resdir;
% % % mkdir(figsdir);
% saveAllFigs
