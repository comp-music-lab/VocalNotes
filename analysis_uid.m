function analysis_uid
    %%
    skip_xcorr = true;

    audiodir = './audio/';
    datadir = './data/';
    dataname = {...
        'GC_Esashi-Oiwake_Song', 'GC_Kuroda-bushi_Song', 'GC_Yagi-bushi_Song',...
        'GJB-T5414R21_Esashi-Oiwake', 'GJB-T5414R26_Kuroda-bushi', 'GJB-T5414R24_Yagi-bushi',...
        'PES_Esashi-Oiwake_Song', 'PES_Kuroda-bushi_Song', 'PES_Yagi-bushi_Song'...
    };
    transcriber = {'YO', 'GC'};
    dataid = '_note_sv.csv';
    
    outputdir = './output/';

    %% ETL
    t_onset = cell(numel(dataname), numel(transcriber));
    t_offset = cell(numel(dataname), numel(transcriber));

    for i=1:numel(dataname)
        for j=1:numel(transcriber)
            dirpath = strcat(datadir, dataname{i}, '/', transcriber{j}, '/');
            listing = dir(dirpath);
            idx = arrayfun(@(l) contains(l.name, dataid), listing);
            filepath = strcat(dirpath, listing(idx).name);
            T = readtable(filepath);

            t_onset{i, j} = T.Var1;
            t_offset{i, j} = T.Var1 + T.Var3;
        end
    end

    %%
    addpath(strcat(userpath, '/lib2/Rcall/'));
    Rlib = 'loe';
    Rpath = 'C:\Program Files\R\R-4.0.2\bin\R.exe';
    Rinit(Rlib, Rpath);

    addpath('./lib/KDE/');
    addpath('./lib/two-sample/');

    c = 5;
    L = 5000;
    nbpobj = nbpindeptest(c, L, 'robust');

    M = 3000;

    freqlimits = [25, 7800];
    bins = 48;
    
    if skip_xcorr
        load('./output/G_ADM.mat', 'G');
    else
        G = cell(numel(dataname), numel(transcriber));
    end
    
    IC = cell(numel(dataname), numel(transcriber));
    IR = cell(numel(dataname), numel(transcriber));
    rho = cell(numel(dataname), numel(transcriber));
    pval = cell(numel(dataname), numel(transcriber));
    BF_01 = cell(numel(dataname), numel(transcriber));

    for i=1:numel(dataname)
        %%
        audiofilepath = strcat(audiodir, dataname{i}, '.wav');
        [s, fs] = audioread(audiofilepath);

        if size(s, 2) == 2
            s = mean(s, 2);
        end

        %%
        y = s(:)';
        t = (0:(numel(y) - 1))./fs;

        for j=1:numel(transcriber)
            if ~skip_xcorr
                %%
                L = numel(t_onset{i, j});
                G_rho = zeros(L, L);
                X = cell(L, 1);
                
                %%
                fw = waitbar(0, 'Wait...');
                tic;
                for l=1:L
                    waitbar(l/L, fw, 'Wait...');
    
                    [~, idx_st] = min(abs(t - t_onset{i, j}(l)));
                    [~, idx_ed] = min(abs(t - t_offset{i, j}(l)));
                    x = y(idx_st:idx_ed);
    
                    cfs = cqt(x, 'SamplingFrequency', fs, 'BinsPerOctave', bins,...
                        'TransformType', 'full', 'FrequencyLimits', freqlimits, 'Window', 'hann');
                    X{l} = cfs.c(1:cfs.NyquistBin, :);
                end
                close(fw);
                toc;
                
                %%
                fw = waitbar(0, 'Wait...');
                tic;
                for l=1:L
                    waitbar(l/L, fw, 'Wait...');
                    X2 = sum(real(X{l}.*conj(X{l})), 2);
                    G_rho(l, l) = 1;
                    
                    for k=(l + 1):L
                        Z = X{k};
                        Z2 = sum(real(Z.*conj(Z)), 2);
                        
                        R = real(xcorr(X{l}(1, :), Z(1, :)));
                        rho = R./sqrt(X2(1)*Z2(1));
    
                        for m=2:size(X{l}, 1)
                             R = real(xcorr(X{l}(m, :), Z(m, :)));
                             rho = rho + R./sqrt(X2(m)*Z2(m));
                        end
                        rho = rho./size(X{l}, 1);
    
                        G_rho(k, l) = max(rho);
                    end
                end
                close(fw);
                toc;
    
                G{i, j} = G_rho + triu(G_rho', 1);
                save(strcat(outputdir, 'G_ADM.mat'), 'G'); 
            end

            %%
            Rpush('ADM', G{i, j});
            Rrun('Y <- LOE(ADM, p=3, c=0.1, eps=1e-5, maxit=1000, method="BFGS", iniX="auto", report=100, DEL=1, H=0.5)');
            Y = Rpull('Y');

            [H, lnf] = bandwidth_mcmc(Y.X, M);
            p = mvkde(Y.X, H);
            IC{i, j} = -log2(p);
            
            IR{i, j} = IC{i, j}.*0;
            for l=1:numel(IR{i, j})
                IR{i, j}(l) = IC{i, j}(l)./(t_offset{i, j}(l) - t_onset{i, j}(l));
            end

            %%
            f1 = figure(1);
            clf; cla;
            f1.Position = [80, 175, 400, 800];
            subplot(2, 1, 1);
            plot(lnf);
            title([transcriber{j}, ': ', dataname{i}], 'Interpreter', 'none');
            subplot(2, 1, 2);
            scatter3(Y.X(:, 1), Y.X(:, 2), Y.X(:, 3));
            drawnow();

            saveas(f1, strcat(outputdir, transcriber{j}, '_', dataname{i}, '_embedding.png'))
            
            %%
            [S, F, T] = spectrogram(y, hann(4096), 4096*7/8, 4096, fs);
            P = 20.*log10(abs(S));

            f2 = figure(2);
            clf; cla;
            subplot(6, 1, 1:3);
            f2.Position = [500, 120, 1400, 850];
            imagesc(T, F, P);
            axis tight;
            ylim([25, 6000]);
            set(gca, 'YDir', 'normal');
            hold on
            yl = ylim();
            for l=1:numel(t_onset{i, j})
                plot(t_onset{i, j}(l).*[1, 1], yl, '-.m');
            end
            hold off;
            title([transcriber{j}, ': ', dataname{i}], 'Interpreter', 'none');
            subplot(6, 1, 4);
            for l=1:numel(t_onset{i, j})
                plot([t_onset{i, j}(l), t_offset{i, j}(l)], IC{i, j}(l).*[1, 1], 'Color', [0, 0.4470, 0.7410]);
                hold on
            end
            ylim([min(IC{i, j}) - 1, max(IC{i, j}) + 1]);
            subplot(6, 1, 5);
            plot(IR{i, j});
            hold on
            scatter(1:numel(IR{i, j}), IR{i, j}, 'MarkerEdgeColor', 'm');
            hold off;
            subplot(6, 1, 6);
            for l=1:numel(t_onset{i, j})
                plot([t_onset{i, j}(l), t_offset{i, j}(l)], IR{i, j}(l).*[1, 1], 'Color', [0, 0.4470, 0.7410]);
                hold on
            end
            ylim([min(IR{i, j}) - 1, max(IR{i, j}) + 1]);
            drawnow();

            saveas(f2, strcat(outputdir, transcriber{j}, '_', dataname{i}, '_IR.png'))
        end

        %%
        f3 = figure(3);
        f3.Position = [100, 500, 1200, 400];
        clf; cla;
        
        for j=1:numel(transcriber)
            %%
            %[rho{i, j}, pval{i, j}] = corr(IC{i, j}, 1./(t_offset{i, j} - t_onset{i, j}), 'Type', 'Pearson', 'Tail', 'left');
            [rho{i, j}, pval{i, j}] = corr(IC{i, j}, IR{i, j}, 'Type', 'Pearson');
            %BF_01{i, j} = -log10(testbayescorr(IC{i, j}, IR{i, j}));
            BF_01{i, j} = nbpobj.test(IC{i, j}, IR{i, j});

            %%
            H = bandwidth_mcmc([IC{i, j}, IR{i, j}], 3000);

            support_x = linspace(min(IC{i, j}) - 1, max(IC{i, j}) + 1, 256)';
            support_y = linspace(min(IR{i, j}) - 10, max(IR{i, j}) + 10, 256)';
            X = cell2mat(arrayfun(@(y) [support_x, repmat(y, [256, 1])], support_y, 'UniformOutput', false));

            P = 0;
            for l=1:numel(IC{i, j})
                P_l = mvnpdf(X, [IC{i, j}(l), IR{i, j}(l)], H);
                P = P + P_l;
            end
            P = reshape(P, [256, 256])';
            
            h = bandwidth_lp(IR{i, j});
            p = kde(support_y(:)', IR{i, j}, h);
            [~, idx] = max(p);
            mode = support_y(idx);

            subplot(1, 6*numel(transcriber), 6*(j - 1)+1);
            plot(p, support_y);
            hold on
            scatter(zeros(numel(IR{i, j}), 1), IR{i, j}, 'Marker', '_', 'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 0.2);
            set(gca, 'YTick', []);
            ylim([support_y(1), support_y(end)]);
            hold off;

            subplot(1, 6*numel(transcriber), (6*j - 4):6*j);
            surf(support_x, support_y, P, 'edgecolor', 'none');
            view(0, 90);
            axis tight;
            hold on
            scatter3(IC{i, j}, IR{i, j}, 10000.*ones(numel(IC{i, j}), 1), 'Marker', 'x', 'MarkerEdgeColor', 'y', 'MarkerEdgeAlpha', 0.5);
            xl = xlim();
            plot3(xl, mode.*[1, 1], 10000.*[1, 1], '-.m');
            hold off;

            title([transcriber{j}, ': ', dataname{i}, ' (', num2str(rho{i, j}, '%3.2f'), ', ', num2str(pval{i, j}, '%3.3f'), ', ', num2str(BF_01{i, j}, '%3.3f'), ')'], 'Interpreter', 'none');

            drawnow();
        end

        saveas(f3, strcat(outputdir, dataname{i}, '_ICIR.png'))
    end
end