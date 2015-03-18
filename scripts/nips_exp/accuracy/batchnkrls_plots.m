% figure
% title('Batch NKRLS Predictions')
% hold on
% plot(expNysBat.result.Y)
% % plot(expNysBat.result.Ypred)
% plot(expNysBat.result.Ypred)
% % legend('Ground truth','Incremental','Batch');
% legend('Ground truth','Batch');
% hold off
% 
% figure
% title('Batch NKRLS Weights')
% hold on
% % plot(expNysBat.algo.c)
% plot(expNysBat.algo.c)
% legend('Batch');
% hold off

%% Batch Nystrom performance (train, val, test)

figure
title('Batch Nystrom performance')
hold on    
h = surf(expNysBat.algo.trainPerformance);
alpha(h,0.2)
h = surf(expNysBat.algo.valPerformance);
alpha(h,0.2)
h = surf(expNysBat.algo.testPerformance);
alpha(h,0.2)
hold off
legend('Training','Validation','Test');
view(45,45)


%% Batch Nystrom performance (only train)

figure
tmp = cell2mat(expNysBat.algo.nyMapper.rng);
pcolor(expNysBat.algo.filterParGuesses , tmp(1,:) , expNysBat.algo.trainPerformance)
title({'Batch Nystrom performance';'Training Set'})
ylabel('m')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar;
h.Label.String = 'RMSE';

%%
figure
hold on
title({'Batch Nystrom performance';'Training Set'})
colormap jet
cc=jet(size(expNysBat.algo.nyMapper.rng(1,:),2));    
for i = 1:size(expNysBat.algo.nyMapper.rng(1,:),2)
    plot(expNysBat.algo.filterParGuesses,expNysBat.algo.trainPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(numel(expNysBat.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysBat.algo.nyMapper.rng(1,:) );
h.Label.String = 'm';

%% Batch Nystrom performance (only val)

figure
pcolor(expNysBat.algo.filterParGuesses,tmp(1,:),expNysBat.algo.valPerformance)
title({'Batch Nystrom performance';'Validation Set'})
ylabel('m')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar;
h.Label.String = 'RMSE';

%%
figure
hold on
title({'Batch Nystrom performance';'Validation Set'})
colormap jet
cc=jet(size(expNysBat.algo.nyMapper.rng(1,:),2));    
for i = 1:size(expNysBat.algo.nyMapper.rng(1,:),2)
    plot(expNysBat.algo.filterParGuesses,expNysBat.algo.valPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(numel(expNysBat.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysBat.algo.nyMapper.rng(1,:) );
h.Label.String = 'm';

%% Batch Nystrom performance (only test)

figure
imagesc()
pcolor(expNysBat.algo.filterParGuesses,tmp(1,:),expNysBat.algo.testPerformance)
title({'Batch Nystrom performance';'Test Set'})
ylabel('m')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar;
h.Label.String = 'RMSE';

%%
figure
hold on
title({'Batch Nystrom performance';'Test Set'})
colormap jet
cc=jet(size(expNysBat.algo.nyMapper.rng(1,:),2));    
for i = 1:size(expNysBat.algo.nyMapper.rng(1,:),2)
    plot(expNysBat.algo.filterParGuesses,expNysBat.algo.testPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(numel(expNysBat.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysBat.algo.nyMapper.rng(1,:) );
h.Label.String = 'm';


%% Batch Nystrom performance - with repetitions
if numRep > 1
    
    % Median test error surface
    testErrMed = median(expNysBat.algo.testPerformance,3);
    figure
    surf(mRange , lRange , testErrMed)
    title('Incremental Nystrom, median test performance')
    set(gca,'XScale','lin')
    set(gca,'YScale','log')

    % Mean + sd test error surface
    testErrAvg = mean(expNysBat.algo.testPerformance,3);
    testErrSd = std(testErr,1,3);
    figure
    hold on
    title('Incremental Nystrom, mean +- 2\sigma performance')
    surf(mRange , lRange , testErrAvg)
    h = surf(mRange , lRange , testErrAvg + 2*testErrSd);
    alpha(h,0.2)
    h = surf(mRange , lRange , testErrAvg - 2*testErrSd);
    alpha(h,0.2)
    set(gca,'XScale','lin')
    set(gca,'YScale','log')
    hold off
    
end