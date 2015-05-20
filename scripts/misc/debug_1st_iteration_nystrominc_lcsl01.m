setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;
close all;

resdir = '';

%% Initialization

numRep = 1;
storeFullTrainPerf = 0;
storeFullValPerf = 1;
storeFullTestPerf = 0;
verbose = 0;
saveResult = 0;

%% Storage vars init

    % Load dataset
%     ds = Adult(500,16282,'plusMinusOne');
% ds = YearPredictionMSD(500,51630);
ds = Covertype(522910,58102,'plusMinusOne');

%% 

numRep = 10;
numNysParGuessesVec = [1 2 5];

perfvec = zeros(numRep,numel(numNysParGuessesVec));
normalphavec = zeros(numRep,numel(numNysParGuessesVec));
timevec = zeros(numRep,numel(numNysParGuessesVec));
normMvec= zeros(numRep,numel(numNysParGuessesVec));

for i = 1:numRep
    display([ 'Repetition #', num2str(i)])
    
    for k = 1:numel(numNysParGuessesVec)

        display([ 'Steps: ', num2str(numNysParGuessesVec(k))])
    
        % Incremental Nystrom KRLS

        map = @nystromUniformIncremental;

        numNysParGuesses = numNysParGuessesVec(k);
        filterParGuesses = 1e-10;
        mapParGuesses = 1.1;

        alg = incrementalNkrls(map , 100 , 'numNysParGuesses' , numNysParGuesses ,...
                                'mapParGuesses' , mapParGuesses ,  ...
                                'filterParGuesses', filterParGuesses , ...
                                'verbose' , 0 , ...
                                'storeFullTrainPerf' , storeFullTrainPerf , ...
                                'storeFullValPerf' , storeFullValPerf , ...
                                'storeFullTestPerf' , storeFullTestPerf);

        expNysInc = experiment(alg , ds , 1 , true , saveResult , '' , resdir , 0);
        expNysInc.run();

        perfvec(i,k) = expNysInc.result.perf;
        timevec(i,k) = expNysInc.time.train;
        normalphavec(i,k) = norm(expNysInc.algo.nyMapper.alpha{1});
        normMvec(i,k) = norm(expNysInc.algo.nyMapper.M{1});
    end
end


if numRep > 1

    figure
    hold on
    title('Performance')
    boxplot(perfvec)

    figure
    hold on
    title('Training time')
    boxplot(timevec)

    figure
    hold on
    title('Norm of \alpha')
    boxplot(normalphavec)

    figure
    hold on
    title('Norm of M')
    boxplot(normMvec)

else
    
    figure
    hold on
    title('Performance')
    bar(perfvec)

    figure
    hold on
    title('Training time')
    bar(timevec)

    figure
    hold on
    title('Norm of \alpha')
    bar(normalphavec)

    figure
    hold on
    title('Norm of M')
    bar(normMvec)

end