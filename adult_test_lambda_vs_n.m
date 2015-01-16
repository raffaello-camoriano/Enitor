setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

nRange = 500:500:7000;
lambdaStarArray = [];

for k = 1:size(nRange,2)

%% Dataset initialization

% Load small dataset
ds = Adult(nRange(k),16282,'plusMinusOne');

%% Experiment 1 setup, laplace kernel

ker = @laplaceKernel;
fil = @tikhonov;

alg = krls(ker, fil,  1, 100);

exp = experiment(alg , ds , 1 , true , true , '' , resdir);

exp.run();

lambdaStarArray = [lambdaStarArray exp.result.filterParStar];

end

% Plot lambda star as a function of n
semilogy(nRange,lambdaStarArray);
