function analysis_ioi
    %%
    reffreq = 440;
    outputdir = './output/';
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
    pitch = cell(numel(dataname), numel(transcriber));
    ioi = cell(numel(dataname), numel(transcriber));
    ioiratio = cell(numel(dataname), numel(transcriber));

    for i=1:numel(dataname)
        for j=1:numel(transcriber)
            dirpath = strcat(datadir, dataname{i}, '/', transcriber{j}, '/');
            listing = dir(dirpath);
            idx = arrayfun(@(l) contains(l.name, dataid), listing);
            filepath = strcat(dirpath, listing(idx).name);
            T = readtable(filepath);

            t_onset{i, j} = T.Var1;
            t_offset{i, j} = T.Var1 + T.Var3;
            pitch{i, j} = 1200.*log2(T.Var2./reffreq);

            [ioi{i, j}, ioiratio{i, j}] = helper.h_ioi(t_onset{i, j}, []);
            ioi{i, j} = ioi{i, j}(:);
            ioiratio{i, j} = ioiratio{i, j}(:);
        end
    end
    
    %%{
    %% Plot
    J = 4;
    al = 2.2;
    be = 1.6;
    dist = zeros(numel(dataname), 1);

    for i=1:numel(dataname)
        %% Metrical difference - DTW
        [dist_i, ix, iy] = dtw([t_onset{i, 1}, t_offset{i, 1}]', [t_onset{i, 2}, t_offset{i, 2}]');
        dist(i) = dist_i/numel(ix);
        
        %% Plot
        f = figure(1);
        f.Position = [35, 80, 780, 870];
        clf; cla;

        T = max(max(t_offset{i, 1}), max(t_offset{i, 2}));
        L = ceil(T/J);
        for j=1:J
            x_min = (j - 1)*L;
            x_max = j*L;

            subplot(J, 1, j);
            scatter(t_onset{i, 1}, ones(numel(t_onset{i, 1}), 1), 'Marker', 'x', 'MarkerEdgeColor', '#0072BD');
            hold on
            scatter(t_onset{i, 2}, zeros(numel(t_onset{i, 2}), 1), 'Marker', 'x', 'MarkerEdgeColor', '#D95319');

            for k=1:numel(ix)
                plot([t_onset{i, 1}(ix(k)), t_onset{i, 2}(iy(k))], [1, 0], ':m');
            end
            
            pitch_min = min(min(pitch{i, 1}), min(pitch{i, 2}));
            pitch_max = max(max(pitch{i, 1}), max(pitch{i, 2}));
            pitch_scaled = al.*(1/(pitch_max - pitch_min)).*(pitch{i, 1} - pitch_min) + be;
            for k=1:numel(t_onset{i, 1})
                plot([t_onset{i, 1}(k), t_offset{i, 1}(k)], pitch_scaled(k).*[1, 1], 'Color', '#0072BD');
            end
            pitch_scaled = al.*(1/(pitch_max - pitch_min)).*(pitch{i, 2} - pitch_min) + be;
            for k=1:numel(t_onset{i, 2})
                plot([t_onset{i, 2}(k), t_offset{i, 2}(k)], pitch_scaled(k).*[1, 1], 'Color', '#D95319');
            end

            hold off
            ylim([-0.5, 5]);
            xlim([x_min, x_max]);

            if j == 1
                title(dataname{i}, 'Interpreter', 'none', 'FontSize', 14);
                legend(transcriber, 'FontSize', 9,...
                    'Location', 'none', 'Position', [0.8, 0.94, 0.11, 0.04]);
            end

            ax = gca(f);
            ax.FontSize = 12;
            set(gca,'YTickLabel',[]);
        end
        
        xlabel('Time (second)', 'FontSize', 12);
        saveas(f, strcat(outputdir, dataname{i}, '_dtw.png'));
    end
    %}

    %% Statistical difference of IOI and IOI ratio
    addpath('./lib/KDE/');
    addpath('./lib/two-sample/');
    nbpobj = nbpfittest(1, 500, 'robust');
    priorodds = 1;
    log10bf = zeros(numel(dataname), 2);
    posterior_H0 = zeros(numel(dataname), 2);
    y = linspace(-5, 5, 1024);
    
    for k=1:2
        switch k
            case 1
                D = ioi;
                a = 0;
                b = 8.5;
                suffix = 'ioi';
            case 2
                D = ioiratio;
                a = 0;
                b = 1;
                suffix = 'ioiratio';
        end

        for i=1:numel(dataname)
            lnbf = nbpobj.test(D{i, 1}, D{i, 2});
            [posterior_H0(i, k), ~] = nbpobj.posterior(priorodds, lnbf);
            log10bf(i, k) = lnbf/log(10);
        end
        
        f = figure(1 + k);
        f.Position = [50, 127, 1000, 820];
        clf; cla;

        x = normcdf(y).*(b - a);
        density = zeros(numel(y), 2);
        for i=1:numel(dataname)
            for j=1:2
                X = D{i, j};
                Y = norminv((X - a)./(b - a), 0, 1);
                h = bandwidth_lp(Y);
                density_y = kde(y, Y, h);
                density(:, j) = density_y .* 1./normpdf(norminv((x - a)./(b - a), 0, 1), 0, 1) .* (1/(b - a));
            end
    
            subplot(3, 3, i);
            plot(x, density);
            hold on
            if k == 2
                xticks([1/4, 1/3, 1/2, 2/3, 3/4]);
                xticklabels({'1/4', '1/3', '1/2', '2/3', '3/4'});
                xtickangle(45);
                yl = ylim();
                plot(3/4.*[1, 1], yl, 'LineStyle', '-.', 'Color', 0.05.*[1, 1, 1]);
                plot(2/3.*[1, 1], yl, 'LineStyle', '-.', 'Color', 0.05.*[1, 1, 1]);
                plot(1/2.*[1, 1], yl, 'LineStyle', '-.', 'Color', 0.05.*[1, 1, 1]);
                plot(1/3.*[1, 1], yl, 'LineStyle', '-.', 'Color', 0.05.*[1, 1, 1]);
                plot(1/4.*[1, 1], yl, 'LineStyle', '-.', 'Color', 0.05.*[1, 1, 1]);
                ylim(yl);
            end
           
            titlestr = [...
                {[dataname{i}, ', n = (', num2str(numel(ioi{i, 1})), ', ', num2str(numel(ioi{i, 2})), ')']},...
                {['Posterior prob. = ', num2str(posterior_H0(i, k), '%3.3f')]},...
                {['log10 Bayes factor = ', num2str(log10bf(i, k), '%3.3f')]}...
            ];

            title(titlestr, 'Interpreter', 'none', 'FontSize', 10);
    
            if i == 1
                legend(transcriber, 'FontSize', 9, 'Location', 'northeast');
            end

            ax = gca(f);
            ax.FontSize = 8;
            xlim([a, b]);
            hold off;
        end
    
        saveas(f, strcat(outputdir, suffix, '_nbp-two-sample.png'));
    end
end