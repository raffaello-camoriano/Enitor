alg = rlsTikhonov();
alg.init('gaussian');

trainValSet = dataset('trainValSet.mat');
testSet = dataset('testSet.mat');

exp = experiment('Experiment_1_RLStik');
exp.run(alg , trainValSet , testSet)