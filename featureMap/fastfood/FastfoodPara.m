function para = FastfoodPara(n, d)
% parameters for Fastfood kernel expansions
%
% Input:
%   n - basis number used for Fastfood approximation
%   d - dimension of input pattern
% Output:
%   para: parameters for Fastfood
%     para.B: binary scaling matrix B in Eqn.(7) [Le et al. 2013 ICML]
%     para.G: Gaussian scaling matrix G in Eqn.(7)
%     para.PI: permutation matrix PI in Eqn.(7)
%     para.S: scaling matrix in Eqn.(7)
%
% Ji Zhao@CMU
% 12/19/2013
% zhaoji84@gmail.com
% This file is part of the FastMMD [Zhao & Meng 2015] code.
%
% Reference:
% [1] Q.V. Le, T. Sarlos, A.J. Smola. Fastfood - Approximating Kernel Expansions in Loglinear Time. ICML, 2013.
% [2] Ji Zhao, Deyu Meng. FastMMD: Ensemble of Circular Discrepancy for Efficient Two-Sample Test. Neural Computation, 2015.

% See also FastfoodForKernel.

%% preprocessing parameter
% pad the vectors with zeros until d = 2^l holds.
l = ceil(log2(d));
d = 2^l;
k = ceil(n/d);
n = d * k;

B = cell(k, 1);
G = cell(k, 1);
PI = cell(k, 1);
S = cell(k, 1);
for ii = 1:k
    %% prepare matrix for Fastfood
    B{ii} = randsrc(d, 1, [1 -1]);
    G{ii} = randn(d, 1);
    T = randperm(d);
    PI{ii} = T(:);
    % Chi distribution
    % http://en.wikipedia.org/wiki/Chi_distribution
    % sampling via cumulative distribution function
    T = gammaincinv(rand(d,1), d/2, 'lower');
    T = (T*2).^(1/2);
    S{ii} = T * norm(G{ii}, 'fro')^(-1);
end
S1 = zeros(n, 1);
for ii = 1:k
    S1(((ii-1)*d+1):(ii*d)) = S{ii};
end

para.B = B;
para.G = G;
para.PI = PI;
para.S = S1;