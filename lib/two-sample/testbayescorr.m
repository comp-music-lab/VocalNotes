function BF_10 = testbayescorr(x, y)
    %%
    r = corr(x, y, 'Type', 'Pearson');

    %%
    n = numel(x);
    BF_10 = (n/2)^0.5/gamma(1/2) * integral(@(g) ((1 + g).^((n - 2)/(n - 1))./(1 + (1 - r^2).*g)).^((n - 1)/2) .* g.^(-3/2) .* exp(-n./(2.*g)), 0, Inf);
end

%{
n = 240;
r = 0.5;
A = integral(@(g) (1 + g).^((n - 2)/2) .* (1 + (1 - r^2).*g).^(-(n - 1)/2) .* g.^(-3/2) .* exp(-n./(2.*g)), 0, Inf);
B = integral(@(g) ((1 + g).^((n - 2)/(n - 1))./(1 + (1 - r^2).*g)).^((n - 1)/2) .* g.^(-3/2) .* exp(-n./(2.*g)), 0, Inf);
%}