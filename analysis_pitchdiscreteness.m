function analysis_pitchdiscreteness
    %%
    datadir = './data/';
    dataname = {...
        'GC_Esashi-Oiwake_Song', 'GC_Kuroda-bushi_Song', 'GC_Yagi-bushi_Song',...
        'GJB-T5414R21_Esashi-Oiwake', 'GJB-T5414R26_Kuroda-bushi', 'GJB-T5414R24_Yagi-bushi',...
        'PES_Esashi-Oiwake_Song', 'PES_Kuroda-bushi_Song', 'PES_Yagi-bushi_Song'...
        };
    transcriber = {'YO', 'GC'};
    outputdir = './output/';
    
    reffreq = 440;

    %%
    K = 8;
    discreteness = zeros(numel(dataname), numel(transcriber));

    for i=1:numel(dataname)
        for j=1:numel(transcriber)
            f0filepath = strcat(datadir, dataname{i}, '/', transcriber{j}, '/', dataname{i}, '_f0.csv');
            f0info = readtable(f0filepath);
            t = table2array(f0info(:, 1));
            f0 = table2array(f0info(:, 2));
            f0_cent = 1200.*log2(f0./reffreq);

            notefilepath = strcat(datadir, dataname{i}, '/', transcriber{j}, '/', dataname{i}, '_note_sv.csv');
            noteinfo = readtable(notefilepath);
            t_st = table2array(noteinfo(:, 1));
            t_ed = t_st + table2array(noteinfo(:, 3));
            
            discreteness(i, j) = pitchdiscreteness(f0_cent, t, t_st, t_ed, K);
        end
    end

    %%
    figobj = figure;
    
    for i=1:numel(transcriber)
        scatter(i.*ones(numel(dataname), 1), discreteness(:, i));
        hold on;
    end
    
    for i=1:numel(dataname)
        plot(1:numel(transcriber), discreteness(i, :), 'LineStyle', ':', 'Color', [0.3, 0.3, 0.3], 'LineWidth', 1);
    end
    hold off;

    xlim([0, numel(transcriber) + 1]);
    xticks(1:numel(transcriber));
    xticklabels(transcriber);
    ylabel('Pitch discreteness score', 'FontSize', 12);
    title('Pitch discreteness of the acoustic units for each song', 'FontSize', 14);
    ax = gca(figobj);
    ax.FontSize = 12;

    saveas(figobj, strcat(outputdir, 'pitchdiscreteness.png'));
end

function [discreteness, IR] = pitchdiscreteness(f0_cent, t, t_st, t_ed, K, dlt)
    %%
    if nargin < 6
        dlt = 0;
    end

    H = [];
    L = [];
    f0_cent = f0_cent(:);
    
    addpath('./lib/KDE/');

    %%
    for k=1:numel(t_st)
        [~, idx_st] = min(abs(t - t_st(k)));
        [~, idx_ed] = min(abs(t - t_ed(k)));
    
        f0_cent_k = f0_cent(idx_st:idx_ed);
    
        idx_st_k = find(~isinf(f0_cent_k), 1, 'first');
        while ~isempty(idx_st_k)
            idx_ed_k = find(isinf(f0_cent_k(idx_st_k:end)), 1, 'first') - 1 + idx_st_k - 1;
    
            if isempty(idx_ed_k)
                idx_ed_k = numel(f0_cent_k);
            end
            
            if (idx_ed_k - idx_st_k + 1) > K
                X = f0_cent_k(idx_st_k:idx_ed_k);
                
                if dlt > 0
                    eps = dlt.*(rand(numel(X), 1) - 0.5);
                    X = X + eps;
                end

                H_k = klentropy(X, K);
    
                H(end + 1) = H_k;
                L(end + 1) = t(idx_ed_k) - t(idx_st_k);
            end
    
            idx_st_k = find(~isinf(f0_cent_k(idx_ed_k + 1:end)), 1, 'first') + idx_ed_k;
        end
    end
    
    %%
    w = L./sum(L);
    discreteness = w*H';
    IR = w*(1./L');
end