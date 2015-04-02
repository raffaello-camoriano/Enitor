%% KRLS performance (train, val, test)

if size(expKRLS.algo.valPerformance,1)>1 && size(expKRLS.algo.valPerformance,2)>1 


    figure
    title('KRLS performance')
    hold on    
%     h = surf(expKRLS.algo.filterParGuessesStorage(1,:),expKRLS.algo.mapParGuesses,expKRLS.algo.trainPerformance);
%     set(h,'FaceColor',[1 0 0])   
%     alpha(h,0.4)
    h = surf(expKRLS.algo.filterParGuessesStorage,expKRLS.algo.mapParGuesses,expKRLS.algo.valPerformance);
    set(h,'FaceColor',[0 1 0])   
    alpha(h,0.4)
    % h = surf(expKRLS.algo.filterParGuesses,expKRLS.algo.mapParGuesses,expKRLS.algo.testPerformance);
    % set(h,'FaceColor',[0 0 1])   
    % alpha(h,0.4)
    hold off
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    zlabel('RMSE','fontsize',16)
    set(gca,'XScale','log')
    % legend('Training','Validation','Test');
%     legend('Training','Validation');
    legend('Validation');
    view([45 45])
elseif size(expKRLS.algo.valPerformance,1)>1 && size(expKRLS.algo.valPerformance,2)==1
    figure
    title('KRLS performance')
    hold on    
    plot(expKRLS.algo.mapParGuesses,expKRLS.algo.trainPerformance);
    plot(expKRLS.algo.mapParGuesses,expKRLS.algo.valPerformance);
    % plot(expKRLS.algo.mapParGuesses,expKRLS.algo.testPerformance);
    hold off
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    legend('Training','Validation');    
    set(gca,'XScale','log')
elseif size(expKRLS.algo.valPerformance,1)==1 && size(expKRLS.algo.valPerformance,2)>1
    figure
    title('KRLS performance')
    hold on    
    plot(expKRLS.algo.filterParGuessesStorage,expKRLS.algo.trainPerformance);
    plot(expKRLS.algo.filterParGuessesStorage,expKRLS.algo.valPerformance);
%     plot(expKRLS.algo.filterParGuessesStorage,expKRLS.algo.testPerformance);
    hold off
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    legend('Training','Validation');    
    set(gca,'XScale','log')
end

%% KRLS performance (only train)


if expKRLS.algo.isFilterParGuessesFixed == 1
    figure
    image(expKRLS.algo.testPerformance , 'CdataMapping' , 'scaled')
    title({'KRLS performance';'Training Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    ax = gca;
    ax.XTickLabels = strread(num2str(expKRLS.algo.filterParGuessesStorage(1,:) , '%10.2e\n'),'%s');
    ax.YTickLabels = expKRLS.algo.mapParGuesses;
    h = colorbar('Ticks' ,min(min(expKRLS.algo.testPerformance)): (max(max(expKRLS.algo.trainPerformance)) - min(min(expKRLS.algo.testPerformance))) / 10: max(max(expKRLS.algo.testPerformance)));
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
else
    figure
    pcolor(expKRLS.algo.filterParGuessesStorage,expKRLS.algo.mapParGuesses,expKRLS.algo.trainPerformance)
    title({'KRLS performance';'Training Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    set(gca,'XScale','log')
    h = colorbar;
    h = colorbar('Ticks' , 0:1/(expKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expKRLS.algo.mapParGuesses );
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
end


figure
hold on
title({'KRLS performance';'Training Set'})
colormap jet
cc=jet(expKRLS.algo.numMapParGuesses);    
for i = 1:expKRLS.algo.numMapParGuesses
    plot(expKRLS.algo.filterParGuessesStorage(i,:),expKRLS.algo.trainPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(expKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expKRLS.algo.mapParGuesses );
h.Label.String = '\sigma';
set(h.Label,'fontsize',16);

%% KRLS performance (only val)

if expKRLS.algo.isFilterParGuessesFixed == 1
    figure
    image(expKRLS.algo.testPerformance , 'CdataMapping' , 'scaled')
    title({'KRLS performance';'Validation Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    ax = gca;
    ax.XTickLabels = strread(num2str(expKRLS.algo.filterParGuessesStorage(1,:) , '%10.2e\n'),'%s');
    ax.YTickLabels = expKRLS.algo.mapParGuesses;
    h = colorbar('Ticks' ,min(min(expKRLS.algo.testPerformance)): (max(max(expKRLS.algo.valPerformance)) - min(min(expKRLS.algo.testPerformance))) / 10: max(max(expKRLS.algo.testPerformance)));
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
else
    figure
    pcolor(expKRLS.algo.filterParGuessesStorage,expKRLS.algo.mapParGuesses,expKRLS.algo.valPerformance)
    title({'KRLS performance';'Validation Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    set(gca,'XScale','log')
    h = colorbar;
    h = colorbar('Ticks' , 0:1/(expKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expKRLS.algo.mapParGuesses );
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
end


figure
hold on
title({'KRLS performance';'Validation Set'})
colormap jet
cc=jet(expKRLS.algo.numMapParGuesses);    
for i = 1:expKRLS.algo.numMapParGuesses
    plot(expKRLS.algo.filterParGuessesStorage(i,:),expKRLS.algo.valPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE')
xlabel('\lambda')
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(expKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expKRLS.algo.mapParGuesses );
h.Label.String = '\sigma';
set(h.Label,'fontsize',16);

%% KRLS performance (only test)


if expKRLS.algo.isFilterParGuessesFixed == 1
    figure
    image(expKRLS.algo.testPerformance , 'CdataMapping' , 'scaled')
    title({'KRLS performance';'Test Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    ax = gca;
    ax.XTickLabels = strread(num2str(expKRLS.algo.filterParGuessesStorage(1,:) , '%10.2e\n'),'%s');
    ax.YTickLabels = expKRLS.algo.mapParGuesses;
    h = colorbar('Ticks' ,min(min(expKRLS.algo.testPerformance)): (max(max(expKRLS.algo.testPerformance)) - min(min(expKRLS.algo.testPerformance))) / 10: max(max(expKRLS.algo.testPerformance)));
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
else
    figure
    pcolor(expKRLS.algo.filterParGuessesStorage,expKRLS.algo.mapParGuesses,expKRLS.algo.testPerformance)
    title({'KRLS performance';'Test Set'})
    ylabel('\sigma','fontsize',16)
    xlabel('\lambda','fontsize',16)
    set(gca,'XScale','log')
    h = colorbar;
    h = colorbar('Ticks' , 0:1/(expKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expKRLS.algo.mapParGuesses );
    h.Label.String = 'RMSE';
    set(h.Label,'fontsize',16);
end


figure
hold on
title({'KRLS performance';'Test Set'})
colormap jet
cc=jet(expKRLS.algo.numMapParGuesses);    
for i = 1:expKRLS.algo.numMapParGuesses
    plot(expKRLS.algo.filterParGuessesStorage(i,:),expKRLS.algo.testPerformance(i,:),'color',cc(i,:))
end
ylabel('RMSE','fontsize',16)
xlabel('\lambda','fontsize',16)
set(gca,'XScale','log')
h = colorbar('Ticks' , 0:1/(expKRLS.algo.numMapParGuesses - 1):1 , 'TickLabels', expKRLS.algo.mapParGuesses );
h.Label.String = '\sigma';
set(h.Label,'fontsize',16);