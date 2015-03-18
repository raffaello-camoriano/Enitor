% Results plotting

figure
% plot(mGuesses,cell2mat(expDACKRLS.algo.trainPerformance));
hold on;
h = surf(expDACKRLS.algo.filterParGuesses,mGuesses, cell2mat(squeeze(expDACKRLS.algo.trainPerformance)));
set(h,'FaceColor',[1 0 0])   
alpha(h,0.4)
h = surf(expDACKRLS.algo.filterParGuesses,mGuesses, cell2mat(squeeze(expDACKRLS.algo.valPerformance)));
set(h,'FaceColor',[0 1 0])   
alpha(h,0.4)
h = surf(expDACKRLS.algo.filterParGuesses,mGuesses, cell2mat(squeeze(expDACKRLS.algo.testPerformance)));
set(h,'FaceColor',[0 0 1])   
alpha(h,0.4)
% plot(mGuesses,cell2mat(expDACKRLS.algo.testPerformance));
hold off;
title({ 'Divide & Conquer KRLS' ; 'Performances for Varying # of Splits and \lambda'})
legend('Training perf','Validation perf','Test perf')
xlabel('\lambda')
ylabel('# of Splits')
zlabel('Performance')
set(gca,'XScale','log')
set(gca,'YScale','log')
view(45,45)

%% DACKRLS performance (only train)


if expDACKRLS.algo.isFilterParGuessesFixed == 1
    figure
    image(cell2mat(squeeze(expDACKRLS.algo.trainPerformance)) , 'CdataMapping' , 'scaled')
    title({'DACKRLS performance';'Training Set'})
    ylabel('m','fontsize',16)
    xlabel('\lambda','fontsize',16)
    ax = gca;
    ax.XTickLabels = strread(num2str(expDACKRLS.algo.filterParGuesses , '%10.2e\n'),'%s');
    ax.YTickLabels = expDACKRLS.algo.mGuesses;
    h = colorbar('Ticks' ,min(min(cell2mat(squeeze(expDACKRLS.algo.trainPerformance)))): (max(max(cell2mat(squeeze(expDACKRLS.algo.trainPerformance)))) - min(min(cell2mat(squeeze(expDACKRLS.algo.trainPerformance))))) / 10: max(max(cell2mat(squeeze(expDACKRLS.algo.trainPerformance)))));
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
    NumTicks = expDACKRLS.algo.numMGuesses;
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks))
else
%     figure
%     pcolor(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.mapParGuesses,expDACKRLS.algo.trainPerformance)
%     title({'KRLS performance';'Training Set'})
%     ylabel('\sigma','fontsize',16)
%     xlabel('\lambda','fontsize',16)
%     set(gca,'XScale','log')
%     h = colorbar;
%     h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mapParGuesses );
%     h.Label.String = 'RMSE';
%     set(h.Label,'fontsize',16);
warning('Plot not implemented')
end


figure
hold on
title({'DACKRLS performance';'Training Set'})
colormap jet
cc=jet(expDACKRLS.algo.numMGuesses);    
for i = 1:expDACKRLS.algo.numMGuesses
    plot(expDACKRLS.algo.filterParGuesses,cell2mat(squeeze(expDACKRLS.algo.trainPerformance(i,:,:))),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
% h = colorbar('Ticks' ,min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))): (max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))) - min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance))))) / 10: max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))));

h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mGuesses );
% h = colorbar('Ticks' , min(expDACKRLS.algo.mGuesses):max(expDACKRLS.algo.mGuesses)-min(expDACKRLS.algo.mGuesses)/(expDACKRLS.algo.numMGuesses-1):max(expDACKRLS.algo.mGuesses));
h.Label.String = 'm';
set(h.Label,'fontsize',16);

%% KRLS performance (only val)


if expDACKRLS.algo.isFilterParGuessesFixed == 1
    figure
    image(cell2mat(squeeze(expDACKRLS.algo.valPerformance)) , 'CdataMapping' , 'scaled')
    title({'DACKRLS performance';'Validation Set'})
    ylabel('m','fontsize',16)
    xlabel('\lambda','fontsize',16)
    ax = gca;
    ax.XTickLabels = strread(num2str(expDACKRLS.algo.filterParGuesses , '%10.2e\n'),'%s');
    ax.YTickLabels = expDACKRLS.algo.mGuesses;
    h = colorbar('Ticks' ,min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))): (max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))) - min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance))))) / 10: max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))));
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
    NumTicks = expDACKRLS.algo.numMGuesses;
    L = get(gca,'YLim');
    set(gca,'YTick',linspace(L(1),L(2),NumTicks))
else
%     figure
%     pcolor(expDACKRLS.algo.filterParGuessesStorage,expDACKRLS.algo.mapParGuesses,expDACKRLS.algo.trainPerformance)
%     title({'KRLS performance';'Training Set'})
%     ylabel('\sigma','fontsize',16)
%     xlabel('\lambda','fontsize',16)
%     set(gca,'XScale','log')
%     h = colorbar;
%     h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mapParGuesses );
%     h.Label.String = 'RMSE';
%     set(h.Label,'fontsize',16);
warning('Plot not implemented')
end


figure
hold on
title({'DACKRLS performance';'Validation Set'})
colormap jet
cc=jet(expDACKRLS.algo.numMGuesses);    
for i = 1:expDACKRLS.algo.numMGuesses
    plot(expDACKRLS.algo.filterParGuesses,cell2mat(squeeze(expDACKRLS.algo.valPerformance(i,:,:))),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
% h = colorbar('Ticks' ,min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))): (max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))) - min(min(cell2mat(squeeze(expDACKRLS.algo.valPerformance))))) / 10: max(max(cell2mat(squeeze(expDACKRLS.algo.valPerformance)))));

h = colorbar('Ticks' , 0:1/(expDACKRLS.algo.numMGuesses - 1):1 , 'TickLabels', expDACKRLS.algo.mGuesses );
% h = colorbar('Ticks' , min(expDACKRLS.algo.mGuesses):max(expDACKRLS.algo.mGuesses)-min(expDACKRLS.algo.mGuesses)/(expDACKRLS.algo.numMGuesses-1):max(expDACKRLS.algo.mGuesses));
h.Label.String = 'm';
set(h.Label,'fontsize',16);

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
