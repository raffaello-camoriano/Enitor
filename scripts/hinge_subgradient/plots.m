
if numRep == 1

    %% Plot timing

    hFig = figure;
    set(hFig, 'Position', [0 0 600 300]);
    trainingTimes = [ expKRLS.result.time.train , expDACKRLS.result.time.train , expLandweber.result.time.train , expNuMethod.result.time.train , expNysInc.result.time.train ];
    bar(trainingTimes)
    set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'Landweber' , '\nu method', 'incNKRLS'})
    title('Training & Model Selection Times')
    ylabel('Time (s)')

    hFig = figure;
    set(hFig, 'Position', [0 0 600 300]);
    trainingTimes = [ expKRLS.result.time.test , expDACKRLS.result.time.test , expLandweber.result.time.test , expNuMethod.result.time.test, expNysInc.result.time.test ];
    bar(trainingTimes)
    set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'Landweber' , '\nu method', 'incNKRLS'})
    title('Testing Times')
    ylabel('Time (s)')

    %% Plot best test performances

    hFig = figure;
    set(hFig, 'Position', [0 0 600 300]);
    testPerf = [ expKRLS.result.perf , expDACKRLS.result.perf  , expLandweber.result.perf , expNuMethod.result.perf, expNysInc.result.perf];
    bar(testPerf)
    set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'Landweber' , '\nu method', 'incNKRLS'})
    title('Test Performance')
    ylabel('Relative Error')% Results plotting
    
elseif numRep > 1
    

    %% Plot timing

    figure
%     hFig1 = figure(1)
%     set(hFig1, 'Position', [0 0 600 300]);
    trainingTimes = [ KRLS_cumulative_training_time, DACKRLS_cumulative_training_time, Landweber_cumulative_training_time, NuMethod_cumulative_training_time, NysInc_cumulative_training_time ];
    boxplot(trainingTimes)
    title('Training & Model Selection Times')
    xlabel('Algorithm')
    ylabel('Time (s)')
    set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'Landweber' , 'nu method', 'incNKRLS'})


    figure
%     hFig1 = figure(1)
%     set(hFig1, 'Position', [0 0 600 300]);
    trainingTimes = [ NuMethod_cumulative_training_time, NysInc_cumulative_training_time ];
    boxplot(trainingTimes)
    title({'Training & Model Selection Times';'Nu method vs. Incremental Nystrom'})
    xlabel('Algorithm')
    ylabel('Time (s)')
    set(gca,'XTickLabel',{'nu method', 'incNKRLS'})

    figure
%     hFig2 = figure(2)
%     set(hFig2, 'Position', [0 0 600 300]);
    testingTimes =  [ KRLS_cumulative_testing_time, DACKRLS_cumulative_testing_time, Landweber_cumulative_testing_time, NuMethod_cumulative_testing_time, NysInc_cumulative_testing_time ];
    boxplot(testingTimes)
    title('Testing Times')
    xlabel('Algorithm')
    ylabel('Time (s)')
    set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'Landweber' , 'nu method', 'incNKRLS'})
    
    %% Plot best test performances

    figure
%     hFig3 = figure(3)
%     set(hFig3, 'Position', [0 0 600 300]);
    testPerf = [ KRLS_cumulative_test_perf, DACKRLS_cumulative_test_perf, Landweber_cumulative_test_perf, NuMethod_cumulative_test_perf, NysInc_cumulative_test_perf ];
    boxplot(testPerf)
    title('Test Performance')
    xlabel('Algorithm')
    ylabel('Relative Error')% Results plotting
    set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'Landweber' , 'nu method', 'incNKRLS'})


    hFig = figure;
    set(hFig, 'Position', [0 0 600 300]);
    bar(median(testPerf))
    set(gca,'XTickLabel',{'KRLS', 'DACKRLS', 'Landweber' , '\nu method', 'incNKRLS'})
    title('Median Test Performance')
    ylabel('Relative Error')% Results plotting
        
end

% if size(alg.valPerformance,1) > 1
%     figure
%     h = surf(alg.trainPerformance);
%     set(h,'FaceColor',[1 0 0],'LineStyle','none');   
%     alpha(h,0.4)
%     hold on
%     title({'Gradient Descent Performance';'Landweber Filter'});
%     h = surf(alg.valPerformance,'LineStyle','none');
%     set(h,'FaceColor',[0 1 0]);   
%     alpha(h,0.4)
%     h = surf(alg.testPerformance,'LineStyle','none');
%     set(h,'FaceColor',[0 0 1]);   
%     alpha(h,0.4)
%     legend('Training','Validation','Test')
%     xlabel('Iteration')
%     ylabel('\sigma')
%     zlabel('error')
%     hold off
% else
%     figure
%     plot(alg.trainPerformance)
%     hold on
%     %title({'Gradient Descent Performance';'Landweber Filter'});
%     plot(alg.valPerformance)
%     %plot(alg.testPerformance)
%     %legend('Training','Validation','Test')
%     legend('Empirical Error','True Error','Test')
%     xlabel('Iteration')
%     ylabel('Error')
% end

% %% 
% figure
% % plot(mGuesses,cell2mat(expDACKRLS.algo.trainPerformance));
% hold on;
% h = surf(expDACKRLS.algo.filterParGuesses,mGuesses, cell2mat(squeeze(expDACKRLS.algo.trainPerformance)));
% set(h,'FaceColor',[1 0 0])   
% alpha(h,0.4)
% h = surf(expDACKRLS.algo.filterParGuesses,mGuesses, cell2mat(squeeze(expDACKRLS.algo.valPerformance)));
% set(h,'FaceColor',[0 1 0])   
% alpha(h,0.4)
% h = surf(expDACKRLS.algo.filterParGuesses,mGuesses, cell2mat(squeeze(expDACKRLS.algo.testPerformance)));
% set(h,'FaceColor',[0 0 1])   
% alpha(h,0.4)
% % plot(mGuesses,cell2mat(expDACKRLS.algo.testPerformance));
% hold off;
% title({ 'Divide & Conquer KRLS' ; 'Performances for Varying # of Splits and \lambda'})
% legend('Training perf','Validation perf','Test perf')
% xlabel('\lambda')
% ylabel('# of Splits')
% zlabel('Performance')
% set(gca,'XScale','log')
% set(gca,'YScale','log')
% view(45,45)
% 
% %% DACKRLS performance (only train)
% 
% 
% if expDACKRLS.algo.isFilterParGuessesFixed == 1
%     figure
%     image(cell2mat(squeeze(expDACKRLS.algo.trainPerformance)) , 'CdataMapping' , 'scaled')
%     title({'DACKRLS performance';'Training Set'})
%     ylabel('m','fontsize',16)
%     xlabel('\lambda','fontsize',16)
%     ax = gca;
%     ax.XTickLabels = strread(num2str(expDACKRLS.algo.filterParGuesses , '%10.2e\n'),'%s');
%     ax.YTickLabels = expDACKRLS.algo.mGuesses;
%     h = colorbar('Ticks' ,min(min(cell2mat(squeeze(expDACKRLS.algo.trainPerformance)))): (max(max(cell2mat(squeeze(expDACKRLS.algo.trainPerformance)))) - min(min(cell2mat(squeeze(expDACKRLS.algo.trainPerformance))))) / 10: max(max(cell2mat(squeeze(expDACKRLS.algo.trainPerformance)))));
%     h.Label.String = 'RMSE';
%     set(h.Label,'fontsize',16);
%     NumTicks = expDACKRLS.algo.numMGuesses;
%     L = get(gca,'YLim');
%     set(gca,'YTick',linspace(L(1),L(2),NumTicks))
% else
% %     figure
% %     pcolor(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.mapParGuesses,expDACKRLS.algo.trainPerformance)
% %     title({'KRLS performance';'Training Set'})
% %     ylabel('\sigma','fontsize',16)
% %     xlabel('\lambda','fontsize',16)
% %     set(gca,'XScale','log')
% %     h = colorbar;
% %     h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mapParGuesses );
% %     h.Label.String = 'RMSE';
% %     set(h.Label,'fontsize',16);
% warning('Plot not implemented')
% end
% 
% 
% figure
% hold on
% title({'DACKRLS performance';'Training Set'})
% colormap jet
% cc=jet(expDACKRLS.algo.numMGuesses);    
% for i = 1:expDACKRLS.algo.numMGuesses
%     plot(expDACKRLS.algo.filterParGuesses,cell2mat(squeeze(expDACKRLS.algo.trainPerformance(i,:,:))),'color',cc(i,:))
% end
% ylabel('RMSE')
% xlabel('\lambda')
% set(gca,'XScale','log')
% % h = colorbar('Ticks' ,min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))): (max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))) - min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance))))) / 10: max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))));
% 
% h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mGuesses );
% % h = colorbar('Ticks' , min(expDACKRLS.algo.mGuesses):max(expDACKRLS.algo.mGuesses)-min(expDACKRLS.algo.mGuesses)/(expDACKRLS.algo.numMGuesses-1):max(expDACKRLS.algo.mGuesses));
% h.Label.String = 'm';
% set(h.Label,'fontsize',16);
% 
% %% KRLS performance (only val)
% 
% 
% if expDACKRLS.algo.isFilterParGuessesFixed == 1
%     figure
%     image(cell2mat(squeeze(expDACKRLS.algo.valPerformance)) , 'CdataMapping' , 'scaled')
%     title({'DACKRLS performance';'Validation Set'})
%     ylabel('m','fontsize',16)
%     xlabel('\lambda','fontsize',16)
%     ax = gca;
%     ax.XTickLabels = strread(num2str(expDACKRLS.algo.filterParGuesses , '%10.2e\n'),'%s');
%     ax.YTickLabels = expDACKRLS.algo.mGuesses;
%     h = colorbar('Ticks' ,min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))): (max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))) - min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance))))) / 10: max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))));
%     h.Label.String = 'RMSE';
%     set(h.Label,'fontsize',16);
%     NumTicks = expDACKRLS.algo.numMGuesses;
%     L = get(gca,'YLim');
%     set(gca,'YTick',linspace(L(1),L(2),NumTicks))
% else
% %     figure
% %     pcolor(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.mapParGuesses,expDACKRLS.algo.trainPerformance)
% %     title({'KRLS performance';'Training Set'})
% %     ylabel('\sigma','fontsize',16)
% %     xlabel('\lambda','fontsize',16)
% %     set(gca,'XScale','log')
% %     h = colorbar;
% %     h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mapParGuesses );
% %     h.Label.String = 'RMSE';
% %     set(h.Label,'fontsize',16);
% warning('Plot not implemented')
% end
% 
% 
% figure
% hold on
% title({'DACKRLS performance';'Validation Set'})
% colormap jet
% cc=jet(expDACKRLS.algo.numMGuesses);    
% for i = 1:expDACKRLS.algo.numMGuesses
%     plot(expDACKRLS.algo.filterParGuesses,cell2mat(squeeze(expDACKRLS.algo.valPerformance(i,:,:))),'color',cc(i,:))
% end
% ylabel('RMSE')
% xlabel('\lambda')
% set(gca,'XScale','log')
% % h = colorbar('Ticks' ,min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))): (max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))) - min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance))))) / 10: max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))));
% 
% h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mGuesses );
% % h = colorbar('Ticks' , min(expDACKRLS.algo.mGuesses):max(expDACKRLS.algo.mGuesses)-min(expDACKRLS.algo.mGuesses)/(expDACKRLS.algo.numMGuesses-1):max(expDACKRLS.algo.mGuesses));
% h.Label.String = 'm';
% set(h.Label,'fontsize',16);

%% KRLS performance (only test)
% 
% 
% if expDACKRLS.algo.isFilterParGuessesFixed == 1
%     figure
%     image(expDACKRLS.algo.testPerformance , 'CdataMapping' , 'scaled')
%     title({'KRLS performance';'Test Set'})
%     ylabel('\sigma','fontsize',16)
%     xlabel('\lambda','fontsize',16)
%     ax = gca;
%     ax.XTickLabels = strread(num2str(expDACKRLS.algo.filterParGuessesStorage(1,:) , '%10.2e\n'),'%s');
%     ax.YTickLabels = expDACKRLS.algo.mapParGuesses;
%     h = colorbar('Ticks' ,min(min(expDACKRLS.algo.testPerformance)): (max(max(expDACKRLS.algo.testPerformance)) - min(min(expDACKRLS.algo.testPerformance))) / 10: max(max(expDACKRLS.algo.testPerformance)));
%     h.Label.String = 'RMSE';
%     set(h.Label,'fontsize',16);
% else
%     figure
%     pcolor(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.mapParGuesses,expDACKRLS.algo.testPerformance)
%     title({'KRLS performance';'Test Set'})
%     ylabel('\sigma','fontsize',16)
%     xlabel('\lambda','fontsize',16)
%     set(gca,'XScale','log')
%     h = colorbar;
%     h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mapParGuesses );
%     h.Label.String = 'RMSE';
%     set(h.Label,'fontsize',16);
% end
% 
% 
% figure
% hold on
% title({'KRLS performance';'Test Set'})
% colormap jet
% cc=jet(expDACKRLS.algo.numMapParGuesses);    
% for i = 1:expDACKRLS.algo.numMapParGuesses
%     plot(expDACKRLS.algo.filterParGuessesStorage(i,:),expDACKRLS.algo.testPerformance(i,:),'color',cc(i,:))
% end
% ylabel('RMSE','fontsize',16)
% xlabel('\lambda','fontsize',16)
% set(gca,'XScale','log')
% h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mapParGuesses );
% h.Label.String = '\sigma';
% set(h.Label,'fontsize',16);
