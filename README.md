Enitor
======

### Scaling up kernel machines.


 /eːˈniː.tor/, \[eːˈniː.tɔr\]

    I climb, ascend


Enitor provides the MATLAB implementation of several large-scale kernel methods.

**[Hic abundant leones](http://4.bp.blogspot.com/_EQDVIsq4TaM/S4u4AbYdOfI/AAAAAAAAAQY/ub10He7RRbY/s320/Cotton_leones.jpg) - This packge is currently discontinued**


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
