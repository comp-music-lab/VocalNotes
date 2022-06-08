function [W, AC] = cwt_morlet(x, fs, w0, F, gpumode)
    %% CWT: setup
    hfun = @(w0, s, t) 1/sqrt(s) .* exp(-t.^2./(2*s^2)) .* exp(1i*w0.*t./s);
    const = pi^(-0.25)/sqrt(fs);
    
    T = 1./F;
     
    N = length(x);
    dt = 1/fs;
    
    if mod(N, 2) == 0
        M = N/2;
        t_h = (-M:(M - 1)).*dt;
    else
        M = (N - 1)/2;
        t_h = (-M:M).*dt;
    end
        
    %% CWT: execution
    W = zeros(length(T), N);
    AC = zeros(length(T), 3);
    lambda = (4*pi)/(w0 + sqrt(2 + w0^2));
    s = T./lambda;
    M = floor(N/2) + 1;
    
    if gpumode
        x = gpuArray(cast(x, 'single'));
        t_h = gpuArray(cast(t_h, 'single'));
        W = gpuArray(cast(W, 'single'));
        AC = gpuArray(cast(AC, 'single'));
        s = gpuArray(cast(s, 'single'));
        const = gpuArray(cast(const, 'single'));
        w0 = gpuArray(cast(w0, 'single'));
    end

    %%
    X = fft(x);
    
    for i=1:length(s)
        h = const .* hfun(w0, s(i), t_h);

        AC(i, 1) = mean(real(h));
        AC(i, 2) = mean(imag(h));
        h_spectrum = (abs(fft(h)).^2)./N;
        AC(i, 3) = sum(h_spectrum);
        
        % check 0 mean energy condition and unit energy condition
        %if abs(AC(i, 1)) < 1e-4 && abs(AC(i, 2)) < 1e-4 && abs(1 - AC(i, 3)) < 1e-4
            h_shift = [h((M + 1):end) h(1:(M))];
            h_shift = fliplr(h_shift);

            psy = fft(conj(h_shift));
            W(i, :) = ifft(psy.*X);
        %else
        %    W(i, :) = NaN;
        %end
    end
end