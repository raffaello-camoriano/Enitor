function [PHI, THT] = FastfoodForKernel(X, para, sgm, use_spiral)
% Fastfood kernel expansions
% The kernel is Gaussian: k(x,y) = exp(-||x-y||_2^2 / (2*sigma^2)).
%
% Input:
%   X: input patterns, each column is an input pattern
%   para: parameters for Fastfood
%     para.B: binary scaling matrix B in Eqn.(7) [Le et al. 2013 ICML]
%     para.G: Gaussian scaling matrix G in Eqn.(7)
%     para.PI: permutation matrix PI in Eqn.(7)
%     para.S: scaling matrix in Eqn.(7)
%   sgm: bandwidth for Gaussian kernel
%   use_spiral: whether to use Spiral package to peform Walsh-Hadamard transform
%     1: use Spiral, which is much efficient for large-scale data
%     0: use Matlab function fwht
% Output:
%   PHI: feature map in Eqn.(8) [Le et al. 2013 ICML]
%   THT: angles used for feature mapping, V*x*sgm, where V is the same as that in Eqn.(8)
%
% Ji Zhao@CMU
% 12/19/2013
% zhaoji84@gmail.com
% This file is part of the FastMMD [Zhao & Meng 2015] code.
%
% Reference:
% [1] Q.V. Le, T. Sarlos, A.J. Smola. Fastfood - Approximating Kernel Expansions in Loglinear Time. ICML, 2013.
% [2] Ji Zhao, Deyu Meng. FastMMD: Ensemble of Circular Discrepancy for Efficient Two-Sample Test. Neural Computation, 2015.
%
% See also FastfoodPara.

if nargin<4
    use_spiral = 0;
end
if nargin < 3
    sgm = 1;
end

%% preprocessing parameter
% pad the vectors with zeros until d = 2^l holds.
[d0, m]= size(X);
l = ceil(log2(d0));
d = 2^l;
if d == d0
    XX = X;
else
    XX = zeros(d, m);
    XX(1:d0, :) = X; 
end
    
k = numel(para.B);
n = d * k;
THT = zeros(n, m);

if (use_spiral)
    fwht_spiral([1; 1]);
    for ii = 1:k
        B = para.B{ii};
        G = para.G{ii};
        PI = para.PI{ii};
        XX = bsxfun(@times, XX, B);
        T = fwht_spiral(XX);
        T = T(PI, :);
        T = bsxfun(@times, T, G);
        THT(((ii-1)*d+1):(ii*d), :) = fwht_spiral(T);
    end
    S = para.S;
    THT = bsxfun(@times, THT, S/d^(1/2));
else
    for ii = 1:k
        B = para.B{ii};
        G = para.G{ii};
        PI = para.PI{ii};
        XX = bsxfun(@times, XX, B);
        T = fwht(XX, d, 'hadamard');
        T = T(PI, :);
        T = bsxfun(@times, T, G*d);
        THT(((ii-1)*d+1):(ii*d), :) = fwht(T, d, 'hadamard');
    end
    S = para.S;
    THT = bsxfun(@times, THT, S* d^(1/2));
end 

T = THT/sgm;
PHI = [cos(T); sin(T)] * n^(-1/2);