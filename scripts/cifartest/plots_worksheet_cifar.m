%     figure
%     title('Incremental Nystrom performance')
%     hold on    
% %     h = surf(expNysInc.algo.trainPerformance);
% %     alpha(h,0.2)
%     h = surf(res.algo.filterParGuesses,res.algo.mapParGuesses(2,:),res.algo.valPerformance);
% %     h = surf(expNysInc.algo.filterParGuesses,expNysInc.algo.mapParGuesses(1,:),expNysInc.algo.valPerformance);
%     alpha(h,0.2)
%         set(gca, 'XScale', 'log')


for i = 1:numNysParGuesses
    
    hfig = figure
    title({'Incremental Nystrom performance';['m = ' , num2str(expNysInc.algo.mapParGuesses(1,(i-1)*expNysInc.algo.numMapParGuesses+1))]})
    hold on    
%     h = surf(expNysInc.algo.trainPerformance);
%     alpha(h,0.2)
    h = surf(expNysInc.algo.filterParGuesses,...
        expNysInc.algo.mapParGuesses(2,1:20),...
        expNysInc.algo.valPerformance((i-1)*expNysInc.algo.numMapParGuesses+1:i*expNysInc.algo.numMapParGuesses,:));
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
%     M(i) = getframe(gca);
    M(i) = getframe(hfig);
    
end

figure
movie(M,1,3)

movie2gif(M, 'test.gif' , 'DelayTime' , 2 , 'LoopCount' , Inf)

