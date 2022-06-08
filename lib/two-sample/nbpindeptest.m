%% Example
% [lnbf, posterior_H0, posterior_H1] = nbpindeptest(c, L, mode).test(xy);

classdef nbpindeptest < handle
    properties(Constant)
         F = @(x) normcdf(x, 0, 1);
    end
    
    properties
        c, L
        h_mu, h_var
    end
    
    methods
        function obj = nbpindeptest(c, L, mode)
            obj.c = c;
            obj.L = L;
            
            if strcmp('robust', mode)
                obj.h_mu = @median;
                obj.h_var = @(x) mad(x, 1);
            elseif strcmp('normal', mode)
                obj.h_mu = @mean;
                obj.h_var = @(x) sqrt(var(x));
            end
        end
        
        function lnbf = test(obj, x, y)
            xy = [x y];
            
            [record, binidx] = obj.partition(xy, '');
            lnbf = obj.oddsratio(record, binidx);
        end
        
        function [record, binidx] = partition(obj, xy, eps)
            %%
            N = size(xy, 1);

            record_i = N;
            binidx_i = {eps};

            %%
            if length(eps) >= obj.L || N == 1 || 1 == numel(unique(xy(:, 1))) || 1 == numel(unique(xy(:, 2)))
                record = record_i;
                binidx = binidx_i;
            elseif N == 0
                record = [];
                binidx = [];
            else
                z = obj.h_standardization(xy);

                p_x = obj.F(z(:, 1));
                p_y = obj.F(z(:, 2));

                idx_0 = p_x < 0.5 & p_y < 0.5;
                idx_1 = p_x >= 0.5 & p_y < 0.5;
                idx_2 = p_x < 0.5 & p_y >= 0.5;
                idx_3 = p_x >= 0.5 & p_y >= 0.5;

                [record_0, binidx_0] = obj.partition(xy(idx_0, :), strcat(eps, '0'));
                [record_1, binidx_1] = obj.partition(xy(idx_1, :), strcat(eps, '1'));
                [record_2, binidx_2] = obj.partition(xy(idx_2, :), strcat(eps, '2'));
                [record_3, binidx_3] = obj.partition(xy(idx_3, :), strcat(eps, '3'));

                record = [record_i; record_0; record_1; record_2; record_3];
                binidx = [binidx_i; binidx_0; binidx_1; binidx_2; binidx_3];
            end
        end
        
        function lnbf = oddsratio(obj, record, binidx)
            lnbf = 0;
            J = length(record);
            al = ones(4, 1);

            for j=1:J
                N_j = record(j);
                eps_j = binidx{j};
                m = length(eps_j);

                if N_j > 1 || m < obj.L
                    %%
                    idx_0 = strcmp(binidx(:), strcat(eps_j, '0'));
                    idx_1 = strcmp(binidx(:), strcat(eps_j, '1'));
                    idx_2 = strcmp(binidx(:), strcat(eps_j, '2'));
                    idx_3 = strcmp(binidx(:), strcat(eps_j, '3'));

                    N_0 = record(idx_0);
                    N_1 = record(idx_1);
                    N_2 = record(idx_2);
                    N_3 = record(idx_3);

                    if isempty(N_0)
                        N_0 = 0;
                    end

                    if isempty(N_1)
                        N_1 = 0;
                    end

                    if isempty(N_2)
                        N_2 = 0;
                    end

                    if isempty(N_3)
                        N_3 = 0;
                    end

                  %%
                    n = [N_0; N_1; N_2; N_3];
                    lnbf_m = obj.h_oddsratio(n, obj.c, m + 1, al);

                  %%
                    lnbf = lnbf + lnbf_m;
                end
            end
        end
        
        function [posterior_H0, posterior_H1] = posterior(obj, priorodds, lnbf)
            posterior_H0 = 1./(1 + 1./(priorodds .* exp(lnbf)));
            posterior_H1 = 1 - posterior_H0;
        end
        
        function lnbf = h_oddsratio(obj, n, c, m, al)
            al_H1 = al .* c*m^2;
            lnp_H1 = obj.logmarginaldiri(al_H1, n);

            al_H0 = 2 * al_H1(1);
            be = al_H0;

            x = n(1) + n(3);
            lnp_H0_x = obj.logmarginalbeta(x, sum(n), al_H0, be);

            x = n(1) + n(2);
            lnp_H0_y = obj.logmarginalbeta(x, sum(n), al_H0, be);

            lnbf = lnp_H0_x + lnp_H0_y - lnp_H1;
        end

        function lnp = logmarginalbeta(obj, x, n, al, be)
            lnp = gammaln(x + al) + gammaln(n - x + be) - gammaln(n + al + be)...
                + gammaln(al + be) - gammaln(al) - gammaln(be);
        end

        function lnp = logmarginaldiri(obj, al, n)
            lnB_a = sum(gammaln(al + n)) - gammaln(sum(al + n));
            lnB = sum(gammaln(al)) - gammaln(sum(al));

            lnp = lnB_a - lnB;
        end

        function z = h_standardization(obj, xy)
            %%
            mu_x = obj.h_mu(xy(:, 1));
            mu_y = obj.h_mu(xy(:, 2));
            
            %%
            sd_x = obj.h_var(xy(:, 1));
            sd_y = obj.h_var(xy(:, 2));
            
            if sd_x == 0
                sd_x = 1;
            end
            
            if sd_y == 0
                sd_y = 1;
            end
            
            %%
            z = xy;

            z(:, 1) = (xy(:, 1) - mu_x)./sd_x;
            z(:, 2) = (xy(:, 2) - mu_y)./sd_y;
        end
    end
end