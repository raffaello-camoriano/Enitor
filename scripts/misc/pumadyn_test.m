setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = pumadyn;

% Load small dataset
ds = pumadyn(4096,4096, 32 , 'n' , 'h');

%% Experiment 1 setup, Gaussian kernel

map = @gaussianKernel;
fil = @tikhonov;

% alg = krls( map , fil , 'numMapParGuesses' , 10 , 'numFilterParGuesses' , 10 , 'verbose' , 1 , ...
%                         'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);

alg = krls( map , fil , 'mapParGuesses' , linspace(1,5,10) , 'filterParGuesses' , logspace(-5,1,10) , 'verbose' , 1 , ...
                        'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 1);

exp = experiment(alg , ds , 1 , true , true , 'nh' , resdir);

exp.run();
exp.result

%% Experiment 2 setup, Random Fourier Features. Gaussian kernel approximation

% map = @randomFeaturesGaussian;
% fil = @tikhonov;
% 
% alg = rfrls(map , 1000 , fil,  5 , 5 , 2000);
% 
% exp = experiment(alg , ds , 1 , true , true , 'fh' , resdir);
% exp.run();
% 
% exp.result


%% KRLS performance (train, val, test)

figure
title('KRLS performance')
hold on    
h = surf(exp.algo.filterParGuessesStorage,exp.algo.mapParGuesses,exp.algo.trainPerformance);
set(h,'FaceColor',[1 0 0])   
alpha(h,0.4)
h = surf(exp.algo.filterParGuessesStorage,exp.algo.mapParGuesses,exp.algo.valPerformance);
set(h,'FaceColor',[0 1 0])   
alpha(h,0.4)
% h = surf(exp.algo.filterParGuesses,exp.algo.mapParGuesses,exp.algo.testPerformance);
% set(h,'FaceColor',[0 0 1])   
% alpha(h,0.4)
hold off
ylabel('\sigma','fontsize',16)
xlabel('\lambda','fontsize',16)
zlabel('RMSE','fontsize',16)
set(gca,'XScale','log')
% legend('Training','Validation','Test');
legend('Training','Validation');
view([45 -45])

%% KRLS performance (only train)


if exp.algo.isFilterParGuessesFixed == 1
    figure
    image(exp.algo.testPerformance , 'CdataMapping' , 'scaled')
    title({'KRLS performance';'Training Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    ax = gca;
    ax.XTickLabels = strread(num2str(exp.algo.filterParGuessesStorage(1,:) , '%10.2e\n'),'%s');
    ax.YTickLabels = exp.algo.mapParGuesses;
    h = colorbar('Ticks' ,min(min(exp.algo.testPerformance)): (max(max(exp.algo.trainPerformance)) - min(min(exp.algo.testPerformance))) / 10: max(max(exp.algo.testPerformance)));
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
else
    figure
    pcolor(exp.algo.filterParGuessesStorage,exp.algo.mapParGuesses,exp.algo.trainPerformance)
    title({'KRLS performance';'Training Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    set(gca,'XScale','log')
    h = colorbar;
    h = colorbar('Ticks' , 0:1/(exp.algo.numMapParGuesses - 1):1 , 'TickLabels', exp.algo.mapParGuesses );
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
end


figure
hold on
title({'KRLS performance';'Training Set'})
colormap jet
cc=jet(exp.algo.numMapParGuesses);    
for i = 1:exp.algo.numMapParGuesses
    plot(exp.algo.filterParGuessesStorage(i,:),exp.algo.trainPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(exp.algo.numMapParGuesses - 1):1 , 'TickLabels', exp.algo.mapParGuesses );
h.Label.String = '\sigma';
set(h.Label,'fontsize',16);

%% KRLS performance (only val)

if exp.algo.isFilterParGuessesFixed == 1
    figure
    image(exp.algo.testPerformance , 'CdataMapping' , 'scaled')
    title({'KRLS performance';'Validation Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    ax = gca;
    ax.XTickLabels = strread(num2str(exp.algo.filterParGuessesStorage(1,:) , '%10.2e\n'),'%s');
    ax.YTickLabels = exp.algo.mapParGuesses;
    h = colorbar('Ticks' ,min(min(exp.algo.testPerformance)): (max(max(exp.algo.valPerformance)) - min(min(exp.algo.testPerformance))) / 10: max(max(exp.algo.testPerformance)));
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
else
    figure
    pcolor(exp.algo.filterParGuessesStorage,exp.algo.mapParGuesses,exp.algo.valPerformance)
    title({'KRLS performance';'Validation Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    set(gca,'XScale','log')
    h = colorbar;
    h = colorbar('Ticks' , 0:1/(exp.algo.numMapParGuesses - 1):1 , 'TickLabels', exp.algo.mapParGuesses );
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
end


figure
hold on
title({'KRLS performance';'Validation Set'})
colormap jet
cc=jet(exp.algo.numMapParGuesses);    
for i = 1:exp.algo.numMapParGuesses
    plot(exp.algo.filterParGuessesStorage(i,:),exp.algo.valPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(exp.algo.numMapParGuesses - 1):1 , 'TickLabels', exp.algo.mapParGuesses );
h.Label.String = '\sigma';
set(h.Label,'fontsize',16);

%% KRLS performance (only test)


if exp.algo.isFilterParGuessesFixed == 1
    figure
    image(exp.algo.testPerformance , 'CdataMapping' , 'scaled')
    title({'KRLS performance';'Test Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    ax = gca;
    ax.XTickLabels = strread(num2str(exp.algo.filterParGuessesStorage(1,:) , '%10.2e\n'),'%s');
    ax.YTickLabels = exp.algo.mapParGuesses;
    h = colorbar('Ticks' ,min(min(exp.algo.testPerformance)): (max(max(exp.algo.testPerformance)) - min(min(exp.algo.testPerformance))) / 10: max(max(exp.algo.testPerformance)));
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
else
    figure
    pcolor(exp.algo.filterParGuessesStorage,exp.algo.mapParGuesses,exp.algo.testPerformance)
    title({'KRLS performance';'Test Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    set(gca,'XScale','log')
    h = colorbar;
    h = colorbar('Ticks' , 0:1/(exp.algo.numMapParGuesses - 1):1 , 'TickLabels', exp.algo.mapParGuesses );
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
end


figure
hold on
title({'KRLS performance';'Test Set'})
colormap jet
cc=jet(exp.algo.numMapParGuesses);    
for i = 1:exp.algo.numMapParGuesses
    plot(exp.algo.filterParGuessesStorage(i,:),exp.algo.testPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE','fontsize',16)
xlabel('\lambda','fontsize',16)
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(exp.algo.numMapParGuesses - 1):1 , 'TickLabels', exp.algo.mapParGuesses );
h.Label.String = '\sigma';
set(h.Label,'fontsize',16);