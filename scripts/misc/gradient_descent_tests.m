setenv('LC_ALL','C');
addpath(genpath('.'));
 
clearAllButBP;

% Set experimental results relative directory name
resdir = 'results';
mkdir(resdir);

%% Dataset initialization

% Load full dataset
%ds = Adult;

% Load small dataset
% ds = Adult(7000,16282,'plusMinusOne');
ds = Adult(4000,16282,'plusMinusOne');


%% Experiment 1 setup, Landweber, Gaussian kernel

map = @gaussianKernel;
fil = @gdesc_square_loss;
maxiter = 5000;


alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , 1:maxiter   , 'verbose' , 0 , ...
                        'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 0);

expLandweber = experiment(alg , ds , 1 , true , true , '' , resdir);

expLandweber.run();
expLandweber.result

%% Plots

if size(alg.valPerformance,1) > 1
    figure
    h = surf(alg.trainPerformance);
    set(h,'FaceColor',[1 0 0],'LineStyle','none');   
    alpha(h,0.4)
    hold on
    title({'Gradient Descent Performance';'Landweber Filter'});
    h = surf(alg.valPerformance,'LineStyle','none');
    set(h,'FaceColor',[0 1 0]);   
    alpha(h,0.4)
    h = surf(alg.testPerformance,'LineStyle','none');
    set(h,'FaceColor',[0 0 1]);   
    alpha(h,0.4)
    legend('Training','Validation','Test')
    xlabel('Iteration')
    ylabel('\sigma')
    zlabel('error')
    hold off
else
    figure
    plot(alg.trainPerformance)
    hold on
    title({'Gradient Descent Performance';'Landweber Filter'});
    plot(alg.valPerformance)
    plot(alg.testPerformance)
    legend('Training','Validation','Test')
    xlabel('Iteration')
    ylabel('Error')
end

%% Experiment 2 setup, nu method, Gaussian kernel

map = @gaussianKernel;
fil = @numethod_square_loss;
maxiter = 5000;

alg = kgdesc( map , fil , 'numMapParGuesses' , 1 , 'filterParGuesses' , 1:maxiter   , 'verbose' , 0 , ...
                        'storeFullTrainPerf' , 1 , 'storeFullValPerf' , 1 , 'storeFullTestPerf' , 0);

expNuMethod = experiment(alg , ds , 1 , true , true , '' , resdir);

expNuMethod.run();
expNuMethod.result
%% Plots

if size(alg.valPerformance,1) > 1
    figure
    h = surf(alg.trainPerformance);
    set(h,'FaceColor',[1 0 0],'LineStyle','none');   
    alpha(h,0.4)
    hold on
    title({'Gradient Descent Performance';'Nu-method Filter'});
    h = surf(alg.valPerformance,'LineStyle','none');
    set(h,'FaceColor',[0 1 0]);   
    alpha(h,0.4)
    h = surf(alg.testPerformance,'LineStyle','none');
    set(h,'FaceColor',[0 0 1]);   
    alpha(h,0.4)
    legend('Training','Validation','Test')
    xlabel('Iteration')
    ylabel('\sigma')
    zlabel('error')
    hold off
else
    figure
    plot(alg.trainPerformance)
    hold on
    title({'Gradient Descent Performance';'Nu-method Filter'});
    plot(alg.valPerformance)
    plot(alg.testPerformance)
    legend('Training','Validation','Test')
    xlabel('Iteration')
    ylabel('Error')
end