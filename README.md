Enitor
======

### Scaling up kernel machines.

![Scaling up!](http://s14.postimg.org/7qmk8bzsx/Haba_Snow_Mountain_Climbing11.jpg)

Enitor is object-oriented large-scale machine learning framework for MATLAB.



Implemented methods:
--------------------

- RLS
- KRLS
- Batch Random Features KRLS
- Batch Nystrom KRLS
- Incremental Nystrom KRLS
- Incremental Random Features KRLS
- Fastfood
- Divide & Conquer KRLS
- nu method
- Landweber iteration
- SVM subgradient descent

Commands to launch an experiment script from command line:
----------------------------------------------------------

matlab -nodesktop -nosplash -r [test-name] &

nohup matlab -nodesktop -nosplash -r [test-name] &

nohup matlab -nodesktop -nosplash -r [test-name] \</dev/null &\>/dev/null &
