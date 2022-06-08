function [H, lnf, B] = bandwidth_mcmc(X, M, lmd, fullcov)
    %% Reference
    % Zhang, X., King, M. L. & Hyndman, R. J. (2006). A Bayesian approach to bandwidth selection for multivariate kernel density estimation. Computational Statistics & Data Analysis, 50, 3009-3031.

    %% Assumptions behind the implementation
    % The Gaussian function is used for a kernel function.

    if nargin < 3
        lmd = 1;
    end
    if nargin < 4
        fullcov = false;
    end

    %%
    D = size(X, 2);
    N = size(X, 1);
    I = 1:N;
    E_D = eye(D);
    ZERO = zeros(1, D);
    
    if fullcov
        idx = find(triu(rand(D)) ~= 0);
    else
        idx = find(eye(D) ~= 0);
    end
    
    E_idx = eye(numel(idx));

    %% Choose an initial bandwidth parameter
    C = [0.01, 0.1, 1, 10, 100, 1000, 10000];
    lnf = zeros(numel(C), 1);

    for i=1:numel(C)
        H = eye(D)./C(i);
        B = inv(chol(H)');

        lnf(i) = h_fulljoint(X, B, lmd, ZERO, E_D, I, N);
    end
    
    [~, idx_max] = max(lnf);
    H_ini = eye(D)./C(idx_max);

    %%
    B_prp = inv(chol(H_ini)');
    lnu = log(rand(M, 1));    
    lnf_t = -Inf;
    B = cell(0, 1);
    lnf = zeros(M, 1);

    for m=1:M
	    if mod(m, 500) == 0
		    fprintf('%s - %d\n', datetime, m);
	    end
    
	    lnf_prp = h_fulljoint(X, B_prp, lmd, ZERO, E_D, I, N);
    
	    if min(0, lnf_prp - lnf_t) > lnu(m)
		    B(end + 1) = {B_prp};
		    lnf_t = lnf_prp;
		    B_seed = B_prp;
	    end
    
	    B_prp(idx) = mvnrnd(B_seed(idx), E_idx, 1);
        while det(B_prp) < 0
            B_prp(idx) = mvnrnd(B_seed(idx), E_idx, 1);
        end

	    lnf(m) = lnf_prp;
    end
    
    %%
    B_mu = 0;
    for i=ceil(numel(B)/2):numel(B)
	    B_mu = B_mu + B{i};
    end
    B_mu = B_mu./(numel(B) - ceil(numel(B)/2) + 1);

    L = inv(B_mu);
    H = L*L';
end

function lnf = h_fulljoint(X, B, lmd, ZERO, E, I, N)
    %%
    lnf = 0;
    N_inf = 0;
    
    %% This part may be sped up by using the FFT-based method.
    for i=1:N
	    D = bsxfun(@minus, X(i, :), X(I ~= i, :))*B;
	    lnf_i = log(sum(mvnpdf(D, ZERO, E)));

	    if ~isinf(lnf_i)
		    lnf = lnf + lnf_i;
	    else
		    N_inf = N_inf + 1;
	    end
    end

    lnf = lnf + (N - N_inf)*(log(det(B)) - (log(N - 1)));

    %%
    lnf = lnf + sum(log(1./(1 + lmd.*B.^2)), 'all');
end