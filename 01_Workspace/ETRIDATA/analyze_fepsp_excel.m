function results = analyze_fepsp_excel(xlsxFile, userOpts)
% Analyze fEPSP features for 30 columns (signals) over 300 ms from an Excel file.
% Outputs a table with: Channel, Stim_ms, Onset_ms, Latency_ms, Peak_ms,
% Amplitude, Slope_20_80, Polarity, SNR, Notes
%
% Usage:
%   results = analyze_fepsp_excel('data.xlsx');
%   results = analyze_fepsp_excel('data.xlsx', struct('baseline_ms',[-20 0]));
%
% Author: GPT-5 Thinking (MATLAB R2018b+ 권장)

%% ---------- Options ----------
opts = struct();
opts.total_ms         = 300;      % 전체 길이(ms)
opts.bp_lo_hz         = 1;        % Bandpass low-cut (Hz)
opts.bp_hi_hz         = 300;      % Bandpass high-cut (Hz)
opts.bp_order         = 4;        % 필터 차수(양방향 filtfilt 적용)
opts.baseline_ms      = [-20 0];  % baseline 윈도우(ms)
opts.search_onset_ms  = [0 50];   % onset 탐색(ms, stim 이후 기준)
opts.search_peak_ms   = [5 30];   % peak 탐색(ms, onset 이후 기준)
opts.k_sigma          = 3;        % onset 임계(k·σ)
opts.min_consec       = 3;        % onset 임계 연속 샘플 개수
opts.derive_sgolay    = true;     % 추가: 최대 기울기 참고 시 Savitzky–Golay 사용
opts.sgolay_k         = 3;        % sgolay order
opts.sg_window_pts    = 11;       % sgolay window length(홀수)

if nargin >= 2 && ~isempty(userOpts)
    fns = fieldnames(userOpts);
    for i=1:numel(fns), opts.(fns{i}) = userOpts.(fns{i}); end
end

%% ---------- Load Excel ----------
T = readtable(xlsxFile);
X = T{:, :};                            % 숫자열만 예상 (30열)
if size(X,2) < 30
    warning('열 수가 30 미만입니다. 현재 %d열을 처리합니다.', size(X,2));
end
[N, C] = size(X);

% 시간축/샘플레이트 추정
t_ms = linspace(0, opts.total_ms, N).'; % 0~300 ms
Fs   = (N-1) / (opts.total_ms/1000);    % Hz
dt   = 1000/Fs;                         % ms/샘플

%% ---------- (Optional) Bandpass Filtering ----------
Xf = X;
try
    % 상한 주파수가 Nyquist를 넘지 않도록 clamp
    nyq = Fs/2;
    hi  = min(opts.bp_hi_hz, nyq*0.95);
    if hi <= opts.bp_lo_hz
        warning('샘플레이트가 낮아 기본 대역통과가 불가 → 스무딩만 적용합니다.');
        Xf = smoothdata(X, 1, 'movmean', max(3, round(0.001*Fs)));
    else
        [b,a] = butter(opts.bp_order/2, [opts.bp_lo_hz hi]/nyq, 'bandpass');
        for ch = 1:C
            Xf(:,ch) = filtfilt(b,a, X(:,ch));
        end
    end
catch
    warning('필터 실패 → 원신호 사용');
    Xf = X;
end

%% ---------- Helper closures ----------
ms2idx = @(ms) max(1, min(N, round(ms/dt) + 1));  % 0ms→1 인덱스 가정

% baseline 인덱스 범위
b0 = ms2idx(opts.baseline_ms(1));
b1 = ms2idx(opts.baseline_ms(2));
if b1 <= b0, b0 = max(1, b1- round(10/dt)); end

% onset/peak 탐색창 길이
on0_off = ms2idx(opts.search_onset_ms(1));
on1_off = ms2idx(opts.search_onset_ms(2));
pk0_rel = ms2idx(opts.search_peak_ms(1));
pk1_rel = ms2idx(opts.search_peak_ms(2));

%% ---------- Result containers ----------
Channel     = (1:C).';
Stim_ms     = nan(C,1);
Onset_ms    = nan(C,1);
Latency_ms  = nan(C,1);
Peak_ms     = nan(C,1);
Amplitude   = nan(C,1);
Slope_20_80 = nan(C,1);
Polarity    = strings(C,1);
SNR         = nan(C,1);
Notes       = strings(C,1);

%% ---------- Core per-channel analysis ----------
for ch = 1:C
    x = Xf(:,ch);

    % baseline 통계
    mu = mean(x(b0:b1));
    sd = std(x(b0:b1));
    if sd == 0, sd = eps; end

    % 자극 아티팩트(Stim) 검출: 기본 0–5 ms 구간에서 최대 |dx/dt|
    s0 = ms2idx(0);
    s1 = ms2idx(5);
    dx = diff(x);
    [~, stim_rel] = max(abs(dx(s0:min(N-1,s1))));
    stim_idx = s0 - 1 + stim_rel;
    if isempty(stim_idx) || stim_idx<1 || stim_idx>N
        % fallback: 전체에서 최대 기울기
        [~, stim_idx] = max(abs(dx));
    end
    stim_ms = (stim_idx-1)*dt;

    % 자극 이후 탐색창(절대 인덱스)
    on_win0 = max(1, stim_idx + on0_off);
    on_win1 = min(N, stim_idx + on1_off);

    % polarity 결정: 자극 후 0–50 ms에서 최대편향 방향
    seg = x(on_win0:on_win1) - mu;
    [minV, imin] = min(seg); [maxV, imax] = max(seg);
    if abs(minV) >= abs(maxV)
        pol = -1; polStr = "negative";
        targetVec = -(x - mu); % 음수 편향을 양수로 뒤집어 onset 탐색에 사용
    else
        pol = +1; polStr = "positive";
        targetVec = +(x - mu);
    end

    % onset 검출: targetVec가 k*sd 초과로 M 연속
    th = opts.k_sigma * sd;
    onset_idx = NaN;
    run = 0;
    for i = on_win0:on_win1
        if targetVec(i) >= th
            run = run + 1;
            if run >= opts.min_consec
                onset_idx = i - (opts.min_consec-1);
                break;
            end
        else
            run = 0;
        end
    end

    note = "";
    if isnan(onset_idx)
        note = note + "Onset not found; ";
        % 그래도 피크는 시도
        onset_idx = on_win0;
    end

    % Peak 탐색: onset 이후 [5,30] ms 상대 윈도우
    pk0 = min(N, onset_idx + pk0_rel);
    pk1 = min(N, onset_idx + pk1_rel);
    xseg = x(pk0:pk1) - mu;
    if pol<0
        [~, ipk_rel] = min(xseg);  % 더 음의 방향
    else
        [~, ipk_rel] = max(xseg);
    end
    peak_idx = pk0 + ipk_rel - 1;

    % Feature 계산
    onset_ms = (onset_idx-1)*dt;
    peak_ms  = (peak_idx-1)*dt;

    amp = abs(x(peak_idx) - mu);
    snr = amp / sd;

    % 20–80% slope: onset–peak 구간에서 20%/80% 레벨 시점 찾기
    seg2 = x(onset_idx:peak_idx);
    if numel(seg2) < 5
        slope2080 = NaN; note = note + "Short onset-peak; ";
    else
        y = pol*(seg2 - mu);                 % polarity 정방향
        y = y - min(y); ymax = max(y);
        if ymax <= 0
            slope2080 = NaN; note = note + "No rise; ";
        else
            y20 = 0.2*ymax; y80 = 0.8*ymax;
            t_local = ((onset_idx:peak_idx)'-1)*dt; % ms
            t20 = interp_time(t_local, y, y20);
            t80 = interp_time(t_local, y, y80);
            if ~isnan(t20) && ~isnan(t80) && t80>t20
                idx20 = max(1, ms2idx(t20) - (onset_idx-1));
                idx80 = max(1, ms2idx(t80) - (onset_idx-1));
                xx = t_local(idx20:idx80);
                yy = (seg2(idx20:idx80));
                p  = polyfit(xx, yy, 1);
                slope2080 = p(1);            % 단위: (신호단위/ms)
            else
                slope2080 = NaN; note = note + "t20/t80 fail; ";
            end
        end
    end

    % 결과 저장
    Channel(ch)     = ch;
    Stim_ms(ch)     = stim_ms;
    Onset_ms(ch)    = onset_ms;
    Latency_ms(ch)  = onset_ms - stim_ms;
    Peak_ms(ch)     = peak_ms;
    Amplitude(ch)   = amp;
    Slope_20_80(ch) = slope2080;
    Polarity(ch)    = polStr;
    SNR(ch)         = snr;
    Notes(ch)       = note;
end

results = table(Channel, Stim_ms, Onset_ms, Latency_ms, Peak_ms, ...
                Amplitude, Slope_20_80, Polarity, SNR, Notes);

% 저장
[outDir, base, ~] = fileparts(xlsxFile);
if isempty(outDir), outDir = "."; end
outFile = fullfile(outDir, base + "_fEPSP_features.xlsx");
writetable(results, outFile);
fprintf('완료: %s\n', outFile);

% 플롯 예시
% plot_fepsp_example(t_ms, Xf(:,1), results(1,:), opts);

end % function

%% ---------- Utilities ----------
function tAt = interp_time(t, y, yLevel)
% 선형보간으로 y==yLevel 시점 추정(상승구간 가정)
idx = find(y >= yLevel, 1, 'first');
if isempty(idx) || idx==1
    tAt = NaN; return;
end
t1 = t(idx-1); t2 = t(idx);
y1 = y(idx-1); y2 = y(idx);
if y2==y1, tAt = NaN; else
    tAt = t1 + (yLevel - y1)*(t2 - t1)/(y2 - y1);
end
end

function plot_fepsp_example(t_ms, x, row, opts)
figure; hold on; grid on;
plot(t_ms, x, 'LineWidth',1);
yL = ylim;

xline(row.Stim_ms,  '--', 'Stim');
xline(row.Onset_ms, '--', 'Onset');
xline(row.Peak_ms,  '--', 'Peak');

title(sprintf('Channel %d | Polarity=%s | Amp=%.3f | Slope(20-80%%)=%.3f', ...
    row.Channel, row.Polarity, row.Amplitude, row.Slope_20_80));
xlabel('Time (ms)'); ylabel('Signal');

% 20–80% 보조선(가능한 경우)
if ~isnan(row.Slope_20_80)
    % 단순 가시화용: onset-peak 직선
    % 실제 20/80% 시점은 본문에서 계산
end
ylim(yL);
end