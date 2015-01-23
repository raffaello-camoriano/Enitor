setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = MNIST;

% Load small dataset
ds = Adult(30000,16282,'plusMinusOne');

% Fixed parameters
fixedsigma = 4.1;

% Set range of m
mRange = 1:50:2000;
nM = size(mRange,2);

% Set range of lambda
lMin = -7;
lMax = 2;
nLambda = 20;
lRange = logspace(lMin,lMax,nLambda);

% Number of experiment repetitions for each parameter combination
numRep = 1;

testErr = zeros(nLambda, nM, numRep);

for j = 1:size(lRange,2)
    l = lRange(j);
    
    for k = 1:size(mRange,2)
        m = mRange(k);

        k

        tmp = [];
        for rep = 1:numRep

            rep

            map = @nystromUniform;
            fil = @tikhonov;

            alg = nrls(map , 1000 , fil,  1 , 1 , m , fixedsigma , l , 0);

            exp = experiment(alg , ds , 1 , true , false , '' , resdir , 0);
            exp.run();

            testErr(j,k,rep) = exp.result.perf;
        end
    end
end

%% Plot results

testErrMed = median(testErr,3);

% Median test error surface
figure
surf(mRange , lRange , testErrMed)
set(gca,'XScale','lin')
set(gca,'YScale','log')

% Test error boxplot
% figure
% boxplot(testErr)
%set(gca,'YScale','log')
