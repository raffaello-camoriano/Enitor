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
ds = MNIST(5000,10000,'plusMinusOne');

% Fixed reg parameters
% fixedlambda = 2.2;
fixedlambda = 10e-07;
fixedsigma = 12.5;

% Set range of m
mRange = 1:50:5000;
numRep = 10;

testErr = [];

for k = 1:size(mRange,2)
    m = mRange(k);
    
    k
    
    tmp = [];
    for rep = 1:numRep

        rep
        
        map = @nystromUniform;
        fil = @tikhonov;

        alg = nrls(map , 1000 , fil,  1 , 1 , m , fixedsigma , fixedlambda , 0);

        exp = experiment(alg , ds , 1 , true , false , '' , resdir , 0);
        exp.run();

        tmp = [tmp ; exp.result.perf];
    end
    testErr = [testErr tmp];
    
end

%% Plot results

% Test error boxplot
figure
boxplot(testErr)
%set(gca,'YScale','log')
