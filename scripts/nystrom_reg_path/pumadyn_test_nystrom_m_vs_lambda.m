setenv('LC_ALL','C');
addpath(genpath('.'));
 
% clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load dataset
ds  = pumadyn(4096,4096, 32 , 'n' , 'h');

% Fixed parameters
fixedsigma = 3.4530;

% Set range of m
mMin = 3000;
mMax = 3000;
nM = 1;
mRange = linspace(mMin,mMax,nM);

% Set range of lambda
lMin = -7;
lMax = 0;
nLambda = 1;
lRange = logspace(lMin,lMax,nLambda);

% Number of experiment repetitions for each parameter combination
numRep = 1;

testErr = zeros(nLambda, nM, numRep);

tic

for j = 1:size(lRange,2)
    l = lRange(j);
    
    j
    
    for k = 1:size(mRange,2)
        m = mRange(k);

        k

        tmp = [];
        for rep = 1:numRep

            
%             ds.reshuffle
            rep

            map = @nystromUniform;
            fil = @tikhonov;

            alg = nrls(map , 1000 , fil,  1 , 1 , 1 , m , fixedsigma , l , 0 , 1 , 1);

            
            exp = experiment(alg , ds , 1 , true , false , '' , resdir , 0);
            exp.run();

            testErr(j,k,rep) = exp.result.perf;
        end
    end
end

toc

% Plot results

% Median test error surface
testErrMed = median(testErr,3);
figure
surf(mRange , lRange , testErrMed)
set(gca,'XScale','lin')
set(gca,'YScale','log')

% Mean + sd test error surface
testErrAvg = mean(testErr,3);
testErrSd = std(testErr,1,3);
figure
hold on
surf(mRange , lRange , testErrAvg)
h = surf(mRange , lRange , testErrAvg + 2*testErrSd);
alpha(h,0.2)
h = surf(mRange , lRange , testErrAvg - 2*testErrSd);
alpha(h,0.2)
set(gca,'XScale','lin')
set(gca,'YScale','log')
hold off