setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

nRange = 50:50:2000;
numRep = 100;
lambdaStarMat = zeros(numRep,size(nRange,2));

for k = 1:size(nRange,2)
    for rep = 1:numRep
        %% Dataset initialization

        % Load small dataset
        ds = Adult(nRange(k),16282,'plusMinusOne');

        %% Experiment 1 setup, laplace kernel

        ker = @laplaceKernel;
        fil = @tikhonov;

        alg = krls(ker, fil,  1, 100);

        exp = experiment(alg , ds , 1 , true , true , '' , resdir);

        exp.run();

        lambdaStarMat(rep,k) = exp.result.filterParStar;
    end
end

%% Plot results
% Plot lambda star as a function of n
semilogy(nRange,mean(lambdaStarMat,1));

%% Boxplot
figure
boxplot(lambdaStarArray)
 set(gca,'YScale','log')
