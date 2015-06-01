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

% incremental nystrom storage vars

incnysTrainTime = [];
batnysTrainTime = [];
m_max = 5000;
numNysParGuesses = 50;
filterParGuesses = 2^(-4);
mapParGuesses = 0.5;

for k = 1:numRep

    display([ 'Repetition #', num2str(k)])
     
    % Load dataset
    ds = cpuSmall(6554,1638);
    
    %% Incremental Nystrom KRLS

    map = @nystromUniformIncremental;


    alg = incrementalNkrls(map , m_max , ...
                            'minRank' , 1 , ...
                            'numNysParGuesses' , numNysParGuesses ,...
                            'mapParGuesses' , mapParGuesses ,  ... 
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf , ...
                            'storeFullTrainTime' , storeFullTrainTime);

    expNysInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
    expNysInc.run();

    incnysTrainTime = [incnysTrainTime ; expNysInc.algo.trainTime'];

    %% Batch Nystrom KRLS

    map = @nystromUniform;
    filter = @tikhonov;

    alg = nrls(map , filter , m_max , ...
                            'numNysParGuesses' , numNysParGuesses ,...
                            'mapParGuesses' , mapParGuesses ,  ... 
                            'filterParGuesses', filterParGuesses , ...
                            'verbose' , 0 , ...
                            'storeFullTrainPerf' , storeFullTrainPerf , ...
                            'storeFullValPerf' , storeFullValPerf , ...
                            'storeFullTestPerf' , storeFullTestPerf , ...
                            'storeFullTrainTime' , storeFullTrainTime);

    expNysBat = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
    expNysBat.run();
   
    batnysTrainTime = [batnysTrainTime ; expNysBat.algo.trainTime'];

end

%% Plots

if numRep == 1
   
    % Plot timing + perf

    figure
    boxplot(expNysInc.algo.trainTime, 'Marker' , 'diamond')
    hold on
    boxplot(expNysBat.algo.trainTime, 'Marker' , 'square')
    ylabel('Time')
    xlabel('Iteration')
    legend('Incremental Nys.','Batch Nys.','Location','northwest')
    set(gca,'YScale','log') 
end

if numRep > 1
    
    % Plot timing + perf

    figure
    errorbar(mean(incnysTrainTime),std(incnysTrainTime,1))
%     plot(median(incnysTrainTime), 'Marker' , 'diamond')
%     boxplot(incnysTrainTime, 'plotstyle' , 'compact')
    hold on
    errorbar(mean(batnysTrainTime),std(batnysTrainTime,1))
    ylabel('Time')
    xlabel('Iteration')
    legend('Incremental Nys.','Batch Nys.','Location','northwest')
%     plot(median(batnysTrainTime), 'Marker' , 'square')
%     boxplot(batnysTrainTime, 'plotstyle' , 'compact')
%     set(gca,'YScale','log') 
    
end

%%
% % 
% % plots
% % 
% % %% Save figures
% figsdir = resdir;
% % % mkdir(figsdir);
% saveAllFigs