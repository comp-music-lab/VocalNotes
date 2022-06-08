function analysis_uid
    %%
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
            notefilepath = strcat(datadir, dataname{i}, '/', transcriber{j}, '/', dataname{i}, dataid);
            T = readtable(notefilepath);
            t_onset{i, j} = table2array(T(:, 1));
            t_offset{i, j} = t_onset{i, j} + table2array(T(:, 3));
        end
    end

    %%
    addpath(strcat(userpath, '/lib2/Rcall/'));
    Rlib = 'loe';
    Rpath = 'C:\Program Files\R\R-4.0.2\bin\R.exe';
    Rinit(Rlib, Rpath);

    addpath('./lib/KDE/');
    numMCMC = 3000;

    load('./output/G_ADM.mat', 'G');
    IC = cell(numel(dataname), numel(transcriber));
    IR = cell(numel(dataname), numel(transcriber));

    for i=1:numel(dataname)
        for j=1:numel(transcriber)
            Rpush('ADM', G{i, j});
            Rrun('Y <- LOE(ADM, p=3, c=0.1, eps=1e-5, maxit=1000, method="BFGS", iniX="auto", report=100, DEL=1, H=0.5)');
            Y = Rpull('Y');
            X = Y.X;
            
            [H, lnf] = bandwidth_mcmc(X, numMCMC);
            p = mvkde(X, H);
            IC{i, j} = -log2(p);

            IR{i, j} = IC{i, j}.*0;
            for l=1:numel(IR{i, j})
                IR{i, j}(l) = IC{i, j}(l)./(t_offset{i, j}(l) - t_onset{i, j}(l));
            end

            figobj = figure(1);
            figobj.Position = [680, 256, 500, 720];
            clf; cla;
            subplot(2, 1, 1);
            plot(lnf);
            title(['Check MCMC convergence: ', dataname{i}, ', ', transcriber{j}], 'Interpreter', 'none');
            subplot(2, 1, 2);
            scatter3(X(:, 1), X(:, 2), X(:, 3));
            drawnow;

            saveas(figobj, strcat(outputdir, transcriber{j}, '_', dataname{i}, '_MCMC-embedding.png'))
        end
    end

    %%
    for j=1:numel(transcriber)
        figobj = figure(1);
        figobj.Position = [90, 150, 800, 800];
        clf; cla;

        for i=1:numel(dataname)
            subplot(3, 3, i);
            scatter(1:numel(IR{i, j}), IR{i, j});
            hold on;
            plot([1, numel(IR{i, j})], median(IR{i, j}).*[1, 1], 'Color', 'm', 'LineWidth', 1.5);
            hold off;
            xlim([1, numel(IR{i, j})]);
            title(['(', transcriber{j}, ') ', dataname{i}], 'Interpreter', 'none', 'FontSize', 11);

            if mod(i, 3) == 1
                ylabel('Information rate (nat/sec.)', 'FontSize', 10);
            end

            if ceil(i/3) == 3
                xlabel('Sound', 'FontSize', 10);
            end
        end

        saveas(figobj, strcat(outputdir, transcriber{j}, '_IR-note.png'));
    end

    %%
    for j=1:numel(transcriber)
        figobj = figure(1);
        figobj.Position = [90, 150, 800, 800];
        clf; cla;

        for i=1:numel(dataname)
            H = bandwidth_mcmc([IC{i, j}, IR{i, j}], 3000);

            support_x = linspace(min(IC{i, j}) - 1, max(IC{i, j}) + 1, 256)';
            support_y = linspace(min(IR{i, j}) - 10, max(IR{i, j}) + 10, 256)';
            X = cell2mat(arrayfun(@(y) [support_x, repmat(y, [256, 1])], support_y, 'UniformOutput', false));

            P = 0;
            for l=1:numel(IC{i, j})
                P = P + mvnpdf(X, [IC{i, j}(l), IR{i, j}(l)], H);
            end
            P = reshape(P, [256, 256])'./numel(IC{i, j});

            subplot(3, 3, i);
            imagesc(support_x, support_y, P);
            set(gca, 'YDir', 'Normal');
            hold on;
            plot([support_x(1), support_x(end)], median(IR{i, j}).*[1, 1], 'Color', 'm', 'LineWidth', 1.5);
            hold off;
            title(['(', transcriber{j}, ') ', dataname{i}], 'Interpreter', 'none', 'FontSize', 11);

            if mod(i, 3) == 1
                ylabel('Information rate (nat/sec.)', 'FontSize', 10);
            end

            if ceil(i/3) == 3
                xlabel('Information content (nat)', 'FontSize', 10);
            end
        end

        saveas(figobj, strcat(outputdir, transcriber{j}, '_IR-IC.png'));
    end
end