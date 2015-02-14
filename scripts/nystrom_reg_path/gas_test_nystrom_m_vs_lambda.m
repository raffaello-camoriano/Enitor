setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load dataset
% ds = Gas(1000,244,'plusMinusOne' , 2);
ds = Gas(1000,586,'plusMinusOne' , 3);

% Fixed parameters
% fixedsigma = 0.0140;
fixedsigma = 0.0174;

% Set range of m
mRange = 1:100:500;
nM = size(mRange,2);

% Set range of lambda
lMin = -7;
lMax = 0;
nLambda = 10;
lRange = logspace(lMin,lMax,nLambda);

% Number of experiment repetitions for each parameter combination
numRep = 3;

testErr = zeros(nLambda, nM, numRep);

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

            alg = nrls(map , 800 , fil,  1 , 1 , m , fixedsigma , l , 0);
            
            exp = experiment(alg , ds , 1 , true , false , '' , resdir , 0);
            exp.run();

            testErr(j,k,rep) = exp.result.perf;
        end
    end
end

%% Plot results
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


