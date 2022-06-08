function analysis_similaritymat
    %%
    audiodir = './audio/';
    datadir = './data/';
    dataname = {...
        'GC_Esashi-Oiwake_Song', 'GC_Kuroda-bushi_Song', 'GC_Yagi-bushi_Song',...
        'GJB-T5414R21_Esashi-Oiwake', 'GJB-T5414R26_Kuroda-bushi', 'GJB-T5414R24_Yagi-bushi',...
        'PES_Esashi-Oiwake_Song', 'PES_Kuroda-bushi_Song', 'PES_Yagi-bushi_Song'...
    };
    transcriber = {'YO', 'GC'};
    dataid = '_note_sv.csv';

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
    G = cell(numel(dataname), numel(transcriber));

    for i=1:numel(dataname)
        %%
        audiofilepath = strcat(audiodir, dataname{i}, '.wav');
        [s, fs] = audioread(audiofilepath);

        if size(s, 2) == 2
            s = mean(s, 2);
        end

        %%
        for j=1:numel(transcriber)
            G{i, j} = signalsimmat(s, fs, t_onset{i, j}, t_offset{i, j});
            save(strcat(outputdir, 'G_ADM.mat'), 'G', 't_onset', 't_offset');
        end
    end
end

function G_rho = signalsimmat(x, fs, t_onset, t_offset)
    %%
    freqlimits = [25, 7800];
    bins = 48;
    t = (0:(numel(x) - 1))./fs;

    %%
    L = numel(t_onset);
    X = cell(L, 1);
    P = cell(L, 1);
    
    fw = waitbar(0, 'Wait...');
    tic;
    for l=1:L
        waitbar(l/L, fw, 'Wait...');

        [~, idx_st] = min(abs(t - t_onset(l)));
        [~, idx_ed] = min(abs(t - t_offset(l)));
        x_l = x(idx_st:idx_ed);

        cfs = cqt(x_l, 'SamplingFrequency', fs, 'BinsPerOctave', bins,...
            'TransformType', 'full', 'FrequencyLimits', freqlimits, 'Window', 'hann');
        X{l} = cfs.c(1:cfs.NyquistBin, :);
        P{l} = sum(real(X{l}.*conj(X{l})), 2);
    end
    close(fw);
    toc;

    %%
    G_rho = zeros(L, L);

    fw = waitbar(0, 'Wait...');
    tic;
    for l=1:L
        waitbar(l/L, fw, 'Wait...');
        G_rho(l, l) = 1;
        
        for k=(l + 1):L
            rho = 0;
            normconst = sqrt(P{l}.*P{k});

            for m=1:size(X{l}, 1)
                 R = real(xcorr(X{l}(m, :), X{k}(m, :)));
                 rho = rho + R./normconst(m);
            end
            rho = rho./size(X{l}, 1);

            G_rho(k, l) = max(rho);
        end
    end
    close(fw);
    toc;

    %%
    G_rho = G_rho + tril(G_rho, -1)';
end