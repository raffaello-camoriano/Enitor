setenv('LC_ALL','C');
addpath(genpath('.'));
% addpath(genpath('~/Repos/libsvm/matlab/'));
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = 'scripts/hinge_subgradient/plots/';
mkdir(resdir);

%% Initialization

numRep =  5;
storeFullTrainPerf = 1;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 0;
saveResult = 0;

%% Storage vars init

trainTime = zeros(numRep,1);
testTime = zeros(numRep,1);
perf = zeros(numRep,1);
% trainErr = zeros(numRep,1);
% valErr = zeros(numRep,1);
trainErr = [];
valErr = [];

for k = 1:numRep

    % Load dataset
    % ds = Adult(7000,16282,'plusMinusOne');
    ds = Adult(3000,1000,'plusMinusOne');
    
    %% Experiment setup, subgradient descent, hinge loss, Gaussian kernel

    map = @gaussianKernel;
    fil = @subgdesc_kernel_hinge_loss;
    maxiter = 15000;
    
    alg = kgdesc( map , fil , ...
        'mapParGuesses' , 6 , ...
        'filterParGuesses' , 1:maxiter   , ...
        'verbose' , 0 , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf);

    expsubgdesc_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
    
    expsubgdesc_kernel_hinge_loss.run();
    expsubgdesc_kernel_hinge_loss.result
    
    trainTime(k) = expsubgdesc_kernel_hinge_loss.time.train;
    testTime(k) = expsubgdesc_kernel_hinge_loss.time.test;
    perf(k) = expsubgdesc_kernel_hinge_loss.result.perf;
    valErr = [ valErr ; expsubgdesc_kernel_hinge_loss.algo.valPerformance];
    trainErr = [ trainErr ; expsubgdesc_kernel_hinge_loss.algo.trainPerformance];
    
%% Experiment setup, stochastic subgradient descent, hinge loss, Gaussian kernel
% 
%     map = @gaussianKernel;
%     fil = @stoc_subgdesc_kernel_hinge_loss;
%     mapParGuesses = expsubgdesc_kernel_hinge_loss.result.mapParStar;
%     maxiter = 7000;
% 
%     alg = Stockgdesc( map , fil , ...

%         'mapParGuesses' , mapParGuesses, ...
%         'filterParGuesses' , 1:maxiter   , ...
%         'verbose' , 0 , ...
%         'storeFullTrainPerf' , storeFullTrainPerf , ...
%         'storeFullValPerf' , storeFullValPerf , ...
%         'storeFullTestPerf' , storeFullTestPerf);
% 
%     expstoc_gdesc_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
% 
%     expstoc_gdesc_kernel_hinge_loss.run();
%     expstoc_gdesc_kernel_hinge_loss.result
% 
%     stoc_gdesc_kernel_hinge_loss_cumulative_training_time(k) = expstoc_gdesc_kernel_hinge_loss .time.train;
%     stoc_gdesc_kernel_hinge_loss_cumulative_testing_time(k) = expstoc_gdesc_kernel_hinge_loss .time.test;
%     stoc_gdesc_kernel_hinge_loss_cumulative_test_perf(k) = expstoc_gdesc_kernel_hinge_loss .result.perf;

end


%% Plot timing

figure
% trainingTimes = [  expsubgdesc_kernel_hinge_loss.result.time.train ];
boxplot(trainTime)
set(gca,'XTickLabel',{ 'Subgr. SVM'})
title('Training & Model Selection Time')
ylabel('Time (s)')

figure
% trainingTimes = [  expsubgdesc_kernel_hinge_loss.result.time.test ];
boxplot(testTime)
set(gca,'XTickLabel',{'Subgr. SVM'})
title('Testing Time')
ylabel('Time (s)')

%% Plot best test performances

figure
% testPerf = [  expsubgdesc_kernel_hinge_loss.result.perf];
boxplot(perf)
set(gca,'XTickLabel',{ 'Subgr. SVM'})
title('Test Performance')
ylabel('Relative Error')

%% Plot training vs validation error over iterations

% Log scale

% figure
% testPerf = [  expsubgdesc_kernel_hinge_loss.result.perf];
axes1 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','log');
box(axes1,'on');
hold on;
% boxplot(trainErr)
% boxplot(valErr)
semilogx(mean(trainErr))
semilogx(mean(valErr))
% set(gca,'XTickLabel',{ 'Subgr. SVM'})
set(gca,'XScale', 'log')
title('Training vs. Validation error')
xlabel('Iteration')
ylabel('Relative Error')
hold off

% linear scale

% figure
% testPerf = [  expsubgdesc_kernel_hinge_loss.result.perf];
axes2 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','linear');
box(axes2,'on');
hold on;
% boxplot(trainErr)
% boxplot(valErr)
plot(mean(trainErr))
plot(mean(valErr))
% set(gca,'XTickLabel',{ 'Subgr. SVM'})
% set(gca,'XScale', 'log')
title('Training vs. Validation Error')
xlabel('Iteration')
ylabel('Relative Error')
hold off

% linear scale

% figure
% testPerf = [  expsubgdesc_kernel_hinge_loss.result.perf];
axes3 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','linear');
box(axes3,'on');
hold on;
% boxplot(trainErr)
% boxplot(valErr)
errorbar(mean(trainErr), std(trainErr))
errorbar(mean(valErr),std(trainErr))
% set(gca,'XTickLabel',{ 'Subgr. SVM'})
% set(gca,'XScale', 'linear')
title('Training vs. Validation Error')
xlabel('Iteration')
ylabel('Relative Error')
hold off

%% Cool graphs

axes3 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','linear');
box(axes3,'on');
title('Training vs. Validation Error')
m = mean(trainErr)';
sd = std(trainErr,0)';
f = [ m'+2*sd' , flipdim(m'-2*sd',2)]; 
a = [1:size(trainErr,2) , size(trainErr,2):-1:1];
hold on;
h = fill( a , f, [7 1 7]/8,'LineStyle','none');
hMean1 = plot(1:size(trainErr,2) , m , 'r' , 'LineWidth',0.5);
alpha(h,0.3);

m2 = mean(valErr)';
sd2 = std(valErr,0)';
f2 = [ m2'+2*sd2' , flipdim(m2'-2*sd2',2)]; 
h = fill([1:size(valErr,2) , size(valErr,2):-1:1] , f2, [1 7 7]/8,'LineStyle','none');
hMean2 = plot(1:size(valErr,2) , m2 , 'b' , 'LineWidth',0.5);
alpha(h,0.3);
xlabel('Iterations')
ylabel('Relative Error')
legend([hMean1,hMean2],'Training','Validation','Location','southeast')




% Log scale

% figure
axes3 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','log');
box(axes3,'on');
title('Training vs. Validation Error')
axis tight
m = mean(trainErr)';
sd = std(trainErr,0)';
f = [ m'+2*sd' , flipdim(m'-2*sd',2)]; 
hold on;
hMean1 = semilogx(1:size(trainErr,2) , m , 'b' , 'LineWidth',1);
h1 = fill([1:size(trainErr,2) , size(trainErr,2):-1:1] , f, 'blue', ...
    'FaceAlpha', 0.1,'LineStyle','none');
semilogx(1:size(trainErr,2) , m , 'b' , 'LineWidth',1);    
alpha(h1,0.3);

m = mean(valErr)';
sd = std(valErr,0)';
f = [ m'+2*sd' , flipdim(m'-2*sd',2)]; 
hMean2 = semilogx(1:size(valErr,2) , m , 'r' , 'LineWidth',1);
% hold on;
h2 = fill([1:size(valErr,2) , size(valErr,2):-1:1] , f, 'red', ...
    'FaceAlpha', 0.1,'LineStyle','none');
semilogx(1:size(valErr,2) , m , 'r' , 'LineWidth',1);    
alpha(h2,0.3);
legend([hMean1,hMean2],'Training','Validation','Location','southwest')
xlabel('Iterations')
ylabel('Relative Error')

%%
% 
% plots
% 
% %% Save figures
figsdir = resdir;
% % mkdir(figsdir);
% saveAllFigs