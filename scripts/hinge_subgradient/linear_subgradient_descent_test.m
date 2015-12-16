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
storeFullTestPerf = 1;
verbose = 0;
saveResult = 0;

maxiter = 5000;


%% Storage vars init

IGD_trainTime = zeros(numRep,1);
IGD_testTime = zeros(numRep,1);
IGD_perf = zeros(numRep,1);
IGD_trainErr = [];
IGD_valErr = [];
IGD_testErr = [];

SGD_trainTime = zeros(numRep,1);
SGD_testTime = zeros(numRep,1);
SGD_perf = zeros(numRep,1);
SGD_trainErr = [];
SGD_valErr = [];
SGD_testErr = [];

SGD_IGD_trainTime = zeros(numRep,1);
SGD_IGD_testTime = zeros(numRep,1);
SGD_IGD_perf = zeros(numRep,1);
SGD_IGD_trainErr = [];
SGD_IGD_valErr = [];
SGD_IGD_testErr = [];

for k = 1:numRep

    % Load dataset
%     ds = Adult(7000,16282,'plusMinusOne');
    ds = Adult(3000,16282,'plusMinusOne');
    ds.lossFunction = @hingeLoss;
%     ds.lossFunction = @classificationError;
    
    %% IGD, hinge loss

    fil = @IsubGD_primal_hinge_loss;
    maxiter = 7400;
    alg = gdesc( fil , ...
        'filterParGuesses' , 1:maxiter   , ...
        'verbose' , 0 , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf);

    expIGD_linear_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
    
    expIGD_linear_hinge_loss.run();
    expIGD_linear_hinge_loss.result
    
    IGD_trainTime(k) = expIGD_linear_hinge_loss.time.train;
    IGD_testTime(k) = expIGD_linear_hinge_loss.time.test;
    IGD_perf(k) = expIGD_linear_hinge_loss.result.perf;
    IGD_trainErr = [IGD_trainErr ; expIGD_linear_hinge_loss.algo.trainPerformance'];
    IGD_valErr = [IGD_valErr ; expIGD_linear_hinge_loss.algo.valPerformance'];
    IGD_testErr = [IGD_testErr ; expIGD_linear_hinge_loss.algo.testPerformance'];
%     
%     
%     %% SGD, hinge loss
%     fil = @SsubGD_primal_hinge_loss;
%     alg = sgdesc( fil , ...
%         'verbose' , 0 , ...
%         'storeFullTrainPerf' , storeFullTrainPerf , ...
%         'storeFullValPerf' , storeFullValPerf , ...
%         'storeFullTestPerf' , storeFullTestPerf);
% 
%     expSGD_linear_hinge_loss = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
%     
%     expSGD_linear_hinge_loss.run();
%     expSGD_linear_hinge_loss.result
%     
%     SGD_trainTime(k) = expSGD_linear_hinge_loss.time.train;
%     SGD_testTime(k) = expSGD_linear_hinge_loss.time.test;
%     SGD_perf(k) = expSGD_linear_hinge_loss.result.perf;
%     SGD_trainErr = [trainErr ; expSGD_linear_hinge_loss.algo.trainPerformance'];
%     SGD_valErr = [valErr ; expSGD_linear_hinge_loss.algo.valPerformance'];
%     SGD_testErr = [testErr ; expSGD_linear_hinge_loss.algo.testPerformance'];
    
    
    
    %% SGD + IGD, hinge loss
    fil = @SsubGD_primal_hinge_loss;
    alg = sgdesc( fil , ...
        'verbose' , 0 , ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf);

    expSGD_IGD_linear_hinge_loss_1 = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
    expSGD_IGD_linear_hinge_loss_1.run();

    maxiter = 5000;
    fil = @IsubGD_primal_hinge_loss;
    alg = gdesc( fil , ...
        'filterParGuesses' , 1:maxiter   , ...
        'verbose' , 0 , ...
        'initialWeights' , expSGD_IGD_linear_hinge_loss_1.algo.c, ...
        'storeFullTrainPerf' , storeFullTrainPerf , ...
        'storeFullValPerf' , storeFullValPerf , ...
        'storeFullTestPerf' , storeFullTestPerf);

    expSGD_IGD_linear_hinge_loss_2 = experiment(alg , ds , 1 , true , saveResult , '' , resdir);
    
    expSGD_IGD_linear_hinge_loss_2.run();
    expSGD_IGD_linear_hinge_loss_2.result
    
%     SGD_IGD_trainTime(k) = expSGD_IGD_linear_hinge_loss.time.train;
%     SGD_IGD_testTime(k) = expSGD_IGD_linear_hinge_loss.time.test;
%     SGD_IGD_perf(k) = expSGD_IGD_linear_hinge_loss.result.perf;
    SGD_IGD_trainErr = [SGD_IGD_trainErr ; [expSGD_IGD_linear_hinge_loss_1.algo.trainPerformance , expSGD_IGD_linear_hinge_loss_2.algo.trainPerformance'] ];
    SGD_IGD_valErr = [SGD_IGD_valErr ; [expSGD_IGD_linear_hinge_loss_1.algo.valPerformance , expSGD_IGD_linear_hinge_loss_2.algo.valPerformance'] ];
    SGD_IGD_testErr = [SGD_IGD_testErr ; [expSGD_IGD_linear_hinge_loss_1.algo.testPerformance , expSGD_IGD_linear_hinge_loss_2.algo.testPerformance'] ];
    
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

% IGD
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

% SGD + IGD
% Log scale

% figure
axes1 = axes('Parent',figure,'YGrid','on','XGrid','on','XScale','log');
box(axes1,'on');
hold on;
if numRep > 1
    semilogx(mean(SGD_IGD_trainErr))
    semilogx(mean(SGD_IGD_valErr))
else
    semilogx(SGD_IGD_trainErr)
    semilogx(SGD_IGD_valErr)
end
set(gca,'XScale', 'log')
title({'SGD + IGD';'Training vs. Validation error'})
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
% saveAllFigs