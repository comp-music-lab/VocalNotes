function x = h_downsampling(s, fs_s, fs_ds)
    [p, q] = rat(fs_ds / fs_s);
    normFc = .98 / max(p, q);
    order = 256 * max(p, q);
    beta = 12;

    lpFilt = firls(order, [0 normFc normFc 1],[1 1 0 0]);
    lpFilt = lpFilt .* kaiser(order + 1, beta)';
    lpFilt = lpFilt / sum(lpFilt);
    lpFilt = p * lpFilt;
    
    x = resample(s, p, q, lpFilt);
end