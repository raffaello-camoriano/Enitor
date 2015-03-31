% figure
% title('Incremental NKRLS Predictions')
% hold on
% plot(expNysInc.result.Y)
% plot(expNysInc.result.Ypred)
% % plot(expNysBat.result.Ypred)
% % legend('Ground truth','Incremental','Batch');
% legend('Ground truth','Incremental');
% hold off

% figure
% title('Incremental NKRLS Weights')
% hold on
% plot(expNysInc.algo.c)
% % plot(expNysBat.algo.c)
% legend('Incremental');
% hold off

%% Incremental Nystrom performance (train, val, test)

if size(expNysInc.algo.valPerformance,1)>1 && size(expNysInc.algo.valPerformance,2)>1 
    figure
    title('Incremental Nystrom performance')
    hold on    
%     h = surf(expNysInc.algo.trainPerformance);
%     alpha(h,0.2)
    h = surf(expNysInc.algo.filterParGuesses,expNysInc.algo.mapParGuesses(2,:),expNysInc.algo.valPerformance);
%     h = surf(expNysInc.algo.filterParGuesses,expNysInc.algo.mapParGuesses(1,:),expNysInc.algo.valPerformance);
    alpha(h,0.2)
%     h = surf(expNysInc.algo.testPerformance);
%     alpha(h,0.2)
    hold off
%     legend('Training','Validation','Test');
    legend('Validation');
    ylabel('\sigma')
    xlabel('\lambda')
    zlabel('Error')
    set(gca, 'XScale', 'log')
    view(45,45)
    
elseif size(expNysInc.algo.valPerformance,1)>1 && size(expNysInc.algo.valPerformance,2)==1
    figure
    title('Incremental Nystrom performance')
    hold on    
%     plot(expNysInc.algo.mapParGuesses(2,:),expNysInc.algo.trainPerformance);
%     plot(expNysInc.algo.mapParGuesses(2,:),expNysInc.algo.valPerformance);
    plot(expNysInc.algo.mapParGuesses(1,:),expNysInc.algo.valPerformance);
%     plot(expKRLS.algo.mapParGuesses,expNysInc.algo.testPerformance);
    hold off
    ylabel('Error','fontsize',16)
    xlabel('\sigma','fontsize',16)
    legend('Training','Validation');    
    set(gca,'XScale','log')
elseif size(expNysInc.algo.valPerformance,1)==1 && size(expNysInc.algo.valPerformance,2)>1
    figure
    title('Incremental Nystrom performance')
    hold on    
    plot(expKRLS.algo.filterParGuessesStorage,expNysInc.algo.trainPerformance);
    plot(expKRLS.algo.filterParGuessesStorage,expNysInc.algo.valPerformance);
%     plot(expKRLS.algo.filterParGuessesStorage,expNysInc.algo.testPerformance);
    hold off
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    legend('Training','Validation');    
    set(gca,'XScale','log')
end

%% Incremental Nystrom performance (only train)
% 
% figure
% pcolor(expNysInc.algo.filterParGuesses,expNysInc.algo.nyMapper.rng(1,:),expNysInc.algo.trainPerformance)
% title({'Incremental Nystrom performance';'Training Set'})
% ylabel('m')
% xlabel('\lambda')
% set(gca,'XScale','log')
% h = colorbar;
% h.Label.String = 'RMSE';
% 
% %%
% figure
% hold on
% title({'Incremental Nystrom performance';'Training Set'})
% colormap jet
% cc=jet(size(expNysInc.algo.nyMapper.rng(1,:),2));    
% for i = 1:size(expNysInc.algo.nyMapper.rng(1,:),2)
%     plot(expNysInc.algo.filterParGuesses,expNysInc.algo.trainPerformance(i,:),'color',cc(i,:))
% end
% ylabel('RMSE')
% xlabel('\lambda')
% set(gca,'XScale','log')
% h = colorbar('Ticks' , 0:1/(numel(expNysInc.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysInc.algo.nyMapper.rng(1,:) );
% h.Label.String = 'm';
% 
% %% Incremental Nystrom performance (only val)
% 
figure
pcolor(expNysInc.algo.filterParGuesses,expNysInc.algo.nyMapper.rng(1,:),expNysInc.algo.valPerformance)
title({'Incremental Nystrom performance';'Validation Set'})
ylabel('m')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar;
h.Label.String = 'RMSE';

% %%
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
% 
% %% Incremental Nystrom performance (only test)
% 
% figure
% imagesc()
% pcolor(expNysInc.algo.filterParGuesses,expNysInc.algo.nyMapper.rng(1,:),expNysInc.algo.testPerformance)
% title({'Incremental Nystrom performance';'Test Set'})
% ylabel('m')
% xlabel('\lambda')
% set(gca,'XScale','log')
% h = colorbar;
% h.Label.String = 'RMSE';
% 
% %%
% figure
% hold on
% title({'Incremental Nystrom performance';'Test Set'})
% colormap jet
% cc=jet(size(expNysInc.algo.nyMapper.rng(1,:),2));    
% for i = 1:size(expNysInc.algo.nyMapper.rng(1,:),2)
%     plot(expNysInc.algo.filterParGuesses,expNysInc.algo.testPerformance(i,:),'color',cc(i,:))
% end
% ylabel('RMSE')
% xlabel('\lambda')
% set(gca,'XScale','log')
% h = colorbar('Ticks' , 0:1/(numel(expNysInc.algo.nyMapper.rng(1,:))-1):1 , 'TickLabels', expNysInc.algo.nyMapper.rng(1,:) );
% h.Label.String = 'm';
% 
% 
% %% Incremental Nystrom performance - with repetitions
% if numRep > 1
%     
%     % Median test error surface
%     testErrMed = median(expNysInc.algo.testPerformance,3);
%     figure
%     surf(mRange , lRange , testErrMed)
%     title('Incremental Nystrom, median test performance')
%     set(gca,'XScale','lin')
%     set(gca,'YScale','log')
% 
%     % Mean + sd test error surface
%     testErrAvg = mean(expNysInc.algo.testPerformance,3);
%     testErrSd = std(testErr,1,3);
%     figure
%     hold on
%     title('Incremental Nystrom, mean +- 2\sigma performance')
%     surf(mRange , lRange , testErrAvg)
%     h = surf(mRange , lRange , testErrAvg + 2*testErrSd);
%     alpha(h,0.2)
%     h = surf(mRange , lRange , testErrAvg - 2*testErrSd);
%     alpha(h,0.2)
%     set(gca,'XScale','lin')
%     set(gca,'YScale','log')
%     hold off
%     
% end