setenv('LC_ALL','C');
addpath(genpath('.'));
clearAllButBP;
close all;

% Set experimental results relative directory name
resdir = 'scripts/hinge_subgradient/plots/';
mkdir(resdir);


%% Initialization

numRep =  1;
storeFullTrainPerf = 1;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 0;
saveResult = 0;

epochs = 100;

map = @gaussianKernel;
mapParGuesses = 6;


%% Storage vars init

IGD_trainTime = zeros(numRep,1);
IGD_testTime = zeros(numRep,1);
IGD_perf = zeros(numRep,1);
IGD_trainErr = [];
IGD_valErr = [];
IGD_testErr = [];

GD_trainTime = zeros(numRep,1);
GD_testTime = zeros(numRep,1);
GD_perf = zeros(numRep,1);
GD_trainErr = [];
GD_valErr = [];
GD_testErr = [];

for k = 1:numRep

    % Load dataset
%     ds = Adult(7000,16282,'plusMinusOne');
    ds = Adult(3000,3000,'plusMinusOne');
    ds.lossFunction = @hingeLoss;
%     ds.lossFunction = @classificationError;
    

    %% IGD, hinge loss

    fil = @IsubGD_dual_hinge_loss;
    maxiter2 = 2400 * epochs;
    alg = kigdesc( map , fil , ...
        'mapParGuesses' , mapParGuesses , ...
        'numFilterParGuesses' , maxiter2   , ...
        'verbose' , 0 , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf);

    expIGD_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
    
    expIGD_kernel_hinge_loss.run();
    expIGD_kernel_hinge_loss.result
    
    IGD_trainTime(k) = expIGD_kernel_hinge_loss.time.train;
    IGD_testTime(k) = expIGD_kernel_hinge_loss.time.test;
    IGD_perf(k) = expIGD_kernel_hinge_loss.result.perf;
    IGD_trainErr = [IGD_trainErr ; expIGD_kernel_hinge_loss.algo.trainPerformance'];
    IGD_valErr = [IGD_valErr ; expIGD_kernel_hinge_loss.algo.valPerformance'];
    IGD_testErr = [IGD_testErr ; expIGD_kernel_hinge_loss.algo.testPerformance'];
    
    
    %% GD, converging step size, hinge loss
    
    fil = @subGD_dual_hinge_loss;
    alg = kgdesc( map , fil , ...
        'numFilterParGuesses' , epochs   , ...
        'mapParGuesses' , mapParGuesses , ...
        'verbose' , 0 , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf);

    expGD_kernel_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
    
    expGD_kernel_hinge_loss.run();
    expGD_kernel_hinge_loss.result
    
    GD_trainTime(k) = expGD_kernel_hinge_loss.time.train;
    GD_testTime(k) = expGD_kernel_hinge_loss.time.test;
    GD_perf(k) = expGD_kernel_hinge_loss.result.perf;
    GD_trainErr = [GD_trainErr ; expGD_kernel_hinge_loss.algo.trainPerformance'];
    GD_valErr = [GD_valErr ; expGD_kernel_hinge_loss.algo.valPerformance'];
    GD_testErr = [GD_testErr ; expGD_kernel_hinge_loss.algo.testPerformance'];
    
end

%% Plot timing

% figure
% boxplot(trainTime)
% set(gca,'XTickLabel',{ 'Subgr. SVM'})
% title('Training & Model Selection Time')
% ylabel('Time (s)')

% figure
% boxplot(testTime)
% set(gca,'XTickLabel',{'Subgr. SVM'})
% title('Testing Time')
% ylabel('Time (s)')

%% Plot best test performances

% figure
% boxplot(perf)
% set(gca,'XTickLabel',{ 'Subgr. SVM'})
% title('Test Performance')
% ylabel('Relative Error')

%% Plot training vs validation error over iterations

% GD: converging step size
% Log scale

% figure
axes1 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','log');
box(axes1,'on');
hold on;
if numRep > 1
    semilogx(mean(GD_trainErr))
    semilogx(mean(GD_valErr))
else
    semilogx(GD_trainErr)
    semilogx(GD_valErr)
end
set(gca,'XScale', 'log')
title({'GD - converging \eta';'Training vs. Validation error'})
xlabel('Iteration')
ylabel('Relative Error')
legend('Training','Validation')
hold off

% IGD\
% Log scale

% figure
axes1 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','log');
box(axes1,'on');
hold on;
if numRep > 1
    semilogx(mean(IGD_trainErr))
    semilogx(mean(IGD_valErr))
else
    semilogx(IGD_trainErr)
    semilogx(IGD_valErr)
end
set(gca,'XScale', 'log')
title({'IGD';'Training vs. Validation error'})
xlabel('Iteration')
ylabel('Relative Error')
legend('Training','Validation')
hold off

% linear scale

% figure
% axes2 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','linear');
% box(axes2,'on');
% hold on;
% plot(mean(trainErr))
% plot(mean(valErr))
% title('Training vs. Validation Error')
% xlabel('Iteration')
% ylabel('Relative Error')
% hold off

% linear scale

% figure
% axes3 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','linear');
% box(axes3,'on');
% hold on;
% errorbar(mean(trainErr), std(trainErr))
% errorbar(mean(valErr),std(trainErr))
% title('Training vs. Validation Error')
% xlabel('Iteration')
% ylabel('Relative Error')
% hold off

%% Cool graphs

% axes3 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','linear');
% box(axes3,'on');
% title('Training vs. Validation Error')
% m = mean(trainErr)';
% sd = std(trainErr,0)';
% f = [ m'+2*sd' , flipdim(m'-2*sd',2)]; 
% a = [1:size(trainErr,2) , size(trainErr,2):-1:1];
% hold on;
% h = fill( a , f, 'b','LineStyle','none');
% hMean1 = plot(1:size(trainErr,2) , m , 'b' , 'LineWidth',0.5);
% alpha(h,0.3);
% 
% m2 = mean(valErr)';
% sd2 = std(valErr,0)';
% f2 = [ m2'+2*sd2' , flipdim(m2'-2*sd2',2)]; 
% h = fill([1:size(valErr,2) , size(valErr,2):-1:1] , f2, 'r','LineStyle','none');
% hMean2 = plot(1:size(valErr,2) , m2 , 'r' , 'LineWidth',0.5);
% alpha(h,0.3);
% xlabel('Iterations')
% ylabel('Relative Error')
% legend([hMean1,hMean2],'Training','Validation','Location','southeast')
% 
% 
% 
% 
% % Log scale
% 
% % figure
% axes3 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','log');
% box(axes3,'on');
% title('Training vs. Validation Error')
% axis tight
% m = mean(trainErr)';
% sd = std(trainErr,0)';
% f = [ m'+2*sd' , flipdim(m'-2*sd',2)]; 
% hold on;
% hMean1 = semilogx(1:size(trainErr,2) , m , 'b' , 'LineWidth',1);
% h1 = fill([1:size(trainErr,2) , size(trainErr,2):-1:1] , f, 'blue', ...
%     'FaceAlpha', 0.1,'LineStyle','none');
% semilogx(1:size(trainErr,2) , m , 'b' , 'LineWidth',1);    
% alpha(h1,0.3);
% 
% m = mean(valErr)';
% sd = std(valErr,0)';
% f = [ m'+2*sd' , flipdim(m'-2*sd',2)]; 
% hMean2 = semilogx(1:size(valErr,2) , m , 'r' , 'LineWidth',1);
% % hold on;
% h2 = fill([1:size(valErr,2) , size(valErr,2):-1:1] , f, 'red', ...
%     'FaceAlpha', 0.1,'LineStyle','none');
% semilogx(1:size(valErr,2) , m , 'r' , 'LineWidth',1);    
% alpha(h2,0.3);
% legend([hMean1,hMean2],'Training','Validation','Location','southwest')
% xlabel('Iterations')
% ylabel('Relative Error')

%%
% 
% plots
% 
% %% Save figures
figsdir = resdir;
% % mkdir(figsdir);
saveAllFigs