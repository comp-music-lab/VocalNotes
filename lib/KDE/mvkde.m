function p = mvkde(X, H)
    %%
    N = size(X, 1);
    p = zeros(N, 1);
    
    D = size(X, 2);
    E = eye(D);
    ZERO = zeros(1, D);

    %% This part may be sped up by using the FFT-based method.
    B = inv(chol(H)');

    for i=1:N
        D = bsxfun(@minus, X(i, :), X)*B;
        p(i) = log(sum(mvnpdf(D, ZERO, E)));
    end

    p = p + log(det(B)) - (log(N));

    p = exp(p);
end