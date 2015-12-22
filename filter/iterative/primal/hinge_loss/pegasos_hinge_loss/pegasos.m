function [wT,b] = pegasos(X,Y,lamda,k,maxIter,Tolerance)
% Usage: [W,b] = pegasos(X,Y,lamda,k,maxIter)
% Input:
% X: n*d matrix, n=number of examples(instances), d=number of variables (features);
% Y: n*1 vector indicating class labels (1 or -1) for examples in X;
% lamda, k: parameters in Pegasos algorithm (default: lamda=1, k=0.1*n);
% maxIter: maximum number of iterations for W-vector convergence; (default: 10000);
% Tolerance: Allowable tolarance for norm of the differance between W-vectors in
% consecutive iterations, to be used as stopping criterion (default: 10^-6);
% 
% Training stops if either of maxIter or Tolerance condition is satisfied;
% 
% Output:
% W,b: parameters in SVM primal problem:
% 
% min 0.5*(||W||)^2
% s.t. (W'*Xi+b)*yi >= 1, for all i=1,...,n
% yi={1,-1};
% 
% This function is implementation of Pegasos paper for SVM classification problem.
% Paper referance:
% "Pegasos-Primal Estimated sub-Gradient SOlver for SVM"
% By Shwartz, Singer and Srebro : 2007
% 
% Code by:
% Thokare Nitin D

[N,d]=size(X);
if(size(Y,1)~=N)
    fprintf('\nError: Number of elements in X and Y must same\nSee pegasos usage for further help\n');
    return;
end
if(sum(Y~=1 & Y~=-1)>0)
    fprintf('\nError: Y must be 1 or -1\nSee pegasos usage for further help\n');
    return;
end

if(nargin<3 || isempty(lamda)),    lamda=1;  end
if(nargin<4 || isempty(maxIter)),    maxIter=10000;  end
if(nargin<5 || isempty(k)),    k=ceil(0.1*N);  end
if(nargin<6 || isempty(Tolerance)),    Tolerance=10^-6;  end

w=rand(1,size(X,2));
w=w/(sqrt(lamda)*norm(w));
for t=1:maxIter
%     fprintf('\niteration # %d/%d',t,maxIter);
    b=mean(Y-X*w(t,:)');
%     idx=randint(k,1,[1,size(X,1)]);
    idx=randi([1,size(X,1)],k,1);
    At=X(idx,:);
    yt=Y(idx,:);
    idx1=(At*w(t,:)'+b).*yt<1;
    etat=1/(lamda*t);
    w1=(1-etat*lamda)*w(t,:)+(etat/k)*sum(At(idx1,:).*repmat(yt(idx1,:),1,size(At,2)),1);
    w(t+1,:)=min(1,1/(sqrt(lamda)*norm(w1)))*w1;
    if(norm(w(t+1,:)-w(t,:)) < Tolerance)
        break;
    end
end
if(t<maxIter)
    fprintf('\nW converged in %d iterations.',t);
else
    fprintf('\nW not converged in %d iterations.',maxIter);
end
wT=mean(w,1);
% wT=w(end,:);
b=mean(Y-X*wT');
Tr=sum(sign(X*wT'+b)==Y);
F=size(X,1)-Tr;
TrainAccuracy=100*Tr/(Tr+F);
fprintf('\nPegasos Accuracy on Training set = %.4f %%\n',TrainAccuracy);
end
