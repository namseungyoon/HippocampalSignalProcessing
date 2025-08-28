%% == fepsp_from_excel_timecol_no_save_RAWandFILTER.m =====================
% 목적:
%   - 1열=시간(ms), 2~31열=신호(30채널) 엑셀을 읽어 채널별 fEPSP 특징을 추출
%   - "왜/무엇"을 주석으로 상세히 설명
%   - Raw(원신호) + Filtered(전처리 후) 신호를 같은 figure에 함께 오버레이
%   - Stim, Onset, Peak, 20–80% slope(t20/t80), Amplitude 수직선 표시
%   - 파일 저장 없음(엑셀/이미지) — 화면에 그래프만 표시
%
% 핵심 알고리즘(무엇/왜):
%   1) 안전한 필터(safe bandpass 또는 스무딩)
%      - 무엇: 1–300 Hz 대역통과 기본 적용, fs가 낮아 경계가 부적절하면 이동평균으로 자동 대체
%      - 왜 : 드리프트/고주파 잡음을 억제해 자극/특징 검출 안정화
%   2) Stim(자극) = 최대 |dV/dt|
%      - 왜 : 자극 아티팩트는 급변 → 최대 기울기 시점이 견고한 기준
%   3) Baseline(μ, σ) = Stim 이전 [−20, 0] ms
%      - 왜 : 임계(|x−μ| ≥ k·σ)와 SNR(Amp/σ)의 기준
%   4) Polarity 자동 판정 (Stim 이후 [0, 50] ms)
%      - 왜 : fEPSP가 보통 음성이나, 전극/레퍼런스에 따라 양성일 수도 있음
%   5) Onset = |x−μ| ≥ k·σ가 연속 M샘플
%      - 왜 : 단발성 잡음 배제
%   6) Peak/Amplitude = Onset 이후 [5, 30] ms 내 polarity 방향 극값, Amp=|peak−μ|
%   7) Slope(20–80%) = Onset–Peak 구간에서 20%/80% 도달시각(t20/t80) 보간 후, 해당 구간 1차 회귀
%
% 작성: GPT-5 Thinking
% ========================================================================

clear; clc; close all;

%% [A] 입력 파일 선택/지정 — (무엇) 엑셀 읽기, (왜) 사용자 포맷 그대로 사용
cand = {"fEPSP_data.xlsx","fEPSP_data.xlxs","fEPSP_data.xls"}; % 철자 실수 보정
inputFile = "";
for i = 1:numel(cand)
    if isfile(cand{i}), inputFile = cand{i}; break; end
end
if inputFile == ""
    [fname, fpath] = uigetfile({'*.xlsx;*.xls','Excel Files (*.xlsx, *.xls)'}, ...
                               '시간 1열 + 신호 2~31열 엑셀 선택');
    if isequal(fname,0), error('파일을 선택하지 않았습니다.'); end
    inputFile = fullfile(fpath, fname);
end

%% [B] 데이터 로드/검증 — (무엇) double 변환/정렬, (왜) 필터/보간 안정화
T   = readtable(inputFile);
A   = T{:, :};
if size(A,2) < 2, error('최소 2열(시간+신호)이 필요합니다.'); end

t_ms = double(A(:,1));       % 시간(ms) — 반드시 double
X    = double(A(:,2:31));    % 2~31열: 30채널(부족하면 적은 만큼 처리)
[N, C] = size(X);
if C==0, error('분석할 신호가 없습니다. (2~31열)'); end
if C<30, warning('신호 열이 30개 미만입니다. 현재 %d개 열을 처리합니다.', C); end

% NaN/비정상 시간 샘플 제거 및 오름차순 정렬
valid  = isfinite(t_ms);
t_ms   = t_ms(valid);
X      = X(valid,:);
[~,ord]= sort(t_ms,'ascend');
t_ms   = t_ms(ord);
X      = X(ord,:);

% dt/Fs 추정 (fallback 포함)
if numel(t_ms) < 2, error('유효한 시간 샘플이 부족합니다.'); end
dt = median(diff(t_ms),'omitnan');  % ms/샘플
if ~isfinite(dt) || dt <= 0
    % 시간축이 깨졌다면 0~300 ms 균등 가정으로 복구
    t_ms = linspace(0,300,size(X,1)).';
    dt   = median(diff(t_ms));
end
Fs = 1000 / dt;                     % Hz

%% [C] 파라미터 — (왜) fEPSP 관행·잡음 수준에 맞춰 쉽게 조정 가능
bp_lo_hz       = 0.05;        % bandpass 하한(저주파 드리프트 억제)
bp_hi_hz       = 300;      % bandpass 상한(고주파 잡음 억제)
bp_order       = 4;        % Butterworth 차수(양방향 filtfilt)
baseline_win   = [-20 0];  % Stim 이전 baseline 창(ms)
onset_search   = [0 50];   % Stim 이후 onset 탐색창(ms)
peak_rel       = [5 30];   % onset 이후 peak 탐색창(ms)
k_sigma        = 3;        % onset 임계(k·σ)
min_consec     = 3;        % onset 연속 샘플 수
NUM_SHOW       = min(C, C);% 화면에 표시할 채널 수(가독성 위해 6). 전체는 C로.

%% [D] 유틸리티 — 안전 필터/보간/색인
function xf = safe_band_filter(x, lo, hi, fs, order)
    % 무엇: 1~300 Hz bandpass 시도. Nyquist 여유 반영, 실패 시 스무딩 대체.
    % 왜 : 샘플레이트가 낮거나 경계가 부적절하면 필터가 에러/불안정해짐.
    nyq   = fs/2;
    hiAdj = min(hi, 0.95*nyq);
    if ~isfinite(nyq) || nyq<=0 || hiAdj<=lo || hiAdj<=0
        % 이동평균 스무딩(대략 1 ms 창) — 왜: 안전한 최소 전처리
        win = max(3, round(0.001*fs));
        if mod(win,2)==0, win = win+1; end
        xf  = movmean(x, win, 1, 'omitnan');
        return;
    end
    try
        [b,a] = butter(order/2, [lo hiAdj]/nyq, 'bandpass');
        xf = filtfilt(b,a,x);
    catch
        win = max(3, round(0.001*fs));
        if mod(win,2)==0, win = win+1; end
        xf  = movmean(x, win, 1, 'omitnan');
    end
end

function idx = idx_at_time(tvec, tval)
    % 무엇: 절대시간→가장 가까운 인덱스
    % 왜  : 구간 선택/마커 표시 시 시간↔인덱스 매핑 필요
    idx = find(tvec>=tval, 1, 'first');
    if isempty(idx), idx = numel(tvec); end
end

function tAt = interp_time_linear(t, y, yLevel)
    % 무엇: y가 상승하며 yLevel에 처음 도달하는 시각(선형보간)
    % 왜  : t20/t80을 서브샘플 정밀도로 추정
    above = find(y >= yLevel, 1, 'first');
    if isempty(above) || above==1, tAt = NaN; return; end
    i  = above;
    t1 = t(i-1);  t2 = t(i);
    y1 = y(i-1);  y2 = y(i);
    if y2==y1, tAt = NaN; else, tAt = t1 + (yLevel - y1)*(t2 - t1)/(y2 - y1); end
end

%% [E] 단일 채널 분석 — Raw와 Filter 모두 반환, 특징 전부 산출
function res = analyze_channel(sig, tvec, fs, dt, bp_lo_hz, bp_hi_hz, bp_order, ...
                               baseline_win, onset_search, peak_rel, k_sigma, min_consec)
    % 1) 필터/스무딩 — (무엇) safe bandpass, (왜) 잡음 억제/안정화
    xf = safe_band_filter(sig, bp_lo_hz, bp_hi_hz, fs, bp_order);

    % 2) Stim — (무엇) 최대 |dV/dt|, (왜) 자극 아티팩트 급변
    dx = [xf(1); diff(xf)];
    [~, stim_idx] = max(abs(dx));
    stim_ms = tvec(stim_idx);

    % 3) Baseline μ, σ — (무엇) Stim 이전 [-20,0]ms, (왜) 임계/SNR 기준
    b0 = idx_at_time(tvec, stim_ms + baseline_win(1));
    b1 = idx_at_time(tvec, stim_ms + baseline_win(2));
    if b1 <= b0, b0 = max(1, b1 - max(3, round(10/dt))); end
    mu = mean(xf(b0:b1), 'omitnan');
    sd = std( xf(b0:b1), 'omitnan'); if sd<=0 || ~isfinite(sd), sd=1e-9; end

    % 4) Polarity — (무엇) Stim 후 [0,50]ms 최대편향 부호, (왜) 음/양 자동 대응
    on0 = idx_at_time(tvec, stim_ms + onset_search(1));
    on1 = idx_at_time(tvec, stim_ms + onset_search(2));
    seg = xf(on0:on1) - mu;
    if isempty(seg), seg=xf-mu; on0=1; on1=numel(xf); end
    [minV,~] = min(seg); [maxV,~] = max(seg); %#ok<ASGLU>
    if abs(minV) >= abs(maxV), pol = -1; polStr="negative"; target = -(xf - mu);
    else,                       pol = +1; polStr="positive"; target = +(xf - mu);
    end

    % 5) Onset — (무엇) |x-μ| ≥ kσ 연속 M 샘플, (왜) 단발 잡음 배제
    th = k_sigma * sd; onset_idx = NaN; run = 0;
    for i = on0:on1
        if target(i) >= th
            run = run + 1;
            if run >= min_consec, onset_idx = i - (min_consec-1); break; end
        else
            run = 0;
        end
    end
    if isnan(onset_idx)
        % 실패 시 창 확장(최대 80 ms)
        on1b = idx_at_time(tvec, stim_ms + 80);
        run = 0;
        for i = on0:on1b
            if target(i) >= th
                run = run + 1;
                if run >= min_consec, onset_idx = i - (min_consec-1); break; end
            else
                run = 0;
            end
        end
    end
    if isnan(onset_idx), onset_idx = on0; end
    onset_ms   = tvec(onset_idx);
    latency_ms = onset_ms - stim_ms;

    % 6) Peak/Amplitude — (무엇) onset 이후 [5,30]ms 극값, Amp=|peak-μ|
    pk0 = idx_at_time(tvec, onset_ms + peak_rel(1));
    pk1 = idx_at_time(tvec, onset_ms + peak_rel(2));
    pk0 = max(pk0, onset_idx); pk1 = max(pk1, pk0+1);
    xseg = xf(pk0:pk1) - mu;
    if isempty(xseg)
        peak_idx = onset_idx;
    else
        % (중요) 조건부 분기로 min/max 선택 — 이전 버그(출력 인수 오류) 방지
        if pol < 0
            [~, ipk] = min(xseg);
        else
            [~, ipk] = max(xseg);
        end
        peak_idx = pk0 + ipk - 1;
    end
    peak_ms   = tvec(peak_idx);
    amplitude = abs(xf(peak_idx) - mu);
    snr       = amplitude / sd;

    % 7) 20–80% Slope — (무엇) t20/t80 보간 후 해당 구간 회귀
    t20 = NaN; t80 = NaN; slope2080 = NaN;
    if peak_idx > onset_idx + 3
        seg2 = xf(onset_idx:peak_idx);
        y = pol*(seg2 - mu); y = y - min(y); ymax = max(y);
        if ymax > 0
            y20 = 0.2*ymax; y80 = 0.8*ymax;
            tloc = tvec(onset_idx:peak_idx);      % 절대 시각
            t20  = interp_time_linear(tloc, y, y20);
            t80  = interp_time_linear(tloc, y, y80);
            if isfinite(t20) && isfinite(t80) && t80>t20
                idx20 = idx_at_time(tvec, t20);
                idx80 = idx_at_time(tvec, t80);
                if idx80 > idx20
                    p = polyfit(tvec(idx20:idx80), xf(idx20:idx80), 1);
                    slope2080 = p(1);             % (신호단위/ms)
                end
            end
        end
    end

    % 결과 패키징(Raw/Filtered/지표/인덱스 모두 반환)
    res = struct('xraw',sig,'xf',xf,'stim_ms',stim_ms,'onset_ms',onset_ms, ...
                 'latency_ms',latency_ms,'peak_ms',peak_ms,'amplitude',amplitude, ...
                 'slope2080',slope2080,'polarity',polStr,'snr',snr,'mu',mu,'sd',sd, ...
                 't20',t20,'t80',t80,'stim_idx',stim_idx,'onset_idx',onset_idx, ...
                 'peak_idx',peak_idx);
end

%% [F] 채널별 분석 + 그래프(저장 없음) — Raw & Filtered 동시 표시
for ch = 1:NUM_SHOW
    sig = X(:,ch);
    res = analyze_channel(sig, t_ms, Fs, dt, bp_lo_hz, bp_hi_hz, bp_order, ...
                          baseline_win, onset_search, peak_rel, k_sigma, min_consec);

    figure('Color','w'); hold on; grid on;

    % (1) Raw 신호 — "무엇" 원자료, "왜" 전처리 영향/왜곡 여부 비교용
    p1 = plot(t_ms, res.xraw, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0); % 회색
    % (2) Filtered 신호 — "무엇" 전처리 결과, "왜" 특징 검출 대상
    p2 = plot(t_ms, res.xf, 'LineWidth', 1.25);

    % (3) Baseline μ, ±3σ — 임계/SNR 시각화
    yline(res.mu,           '--', 'Baseline \mu');
    yline(res.mu + 3*res.sd, ':', '+3\sigma');
    yline(res.mu - 3*res.sd, ':', '-3\sigma');

    % (4) 주요 시점 세로선
    xline(res.stim_ms,  '--', 'Stim');
    xline(res.onset_ms, '--', 'Onset');
    xline(res.peak_ms,  '--', 'Peak');

    % (5) 특징 마커 (Filtered 파형 기준으로 좌표 표시)
    plot(res.stim_ms,  res.xf(res.stim_idx),  's', 'MarkerSize',7, 'LineWidth',1);
    plot(res.onset_ms, res.xf(res.onset_idx), 'o', 'MarkerSize',8, 'LineWidth',1);
    plot(res.peak_ms,  res.xf(res.peak_idx),  '^', 'MarkerSize',8, 'LineWidth',1);

    % (6) 20–80% slope 구간 + 양 끝점 마커 + Amplitude 수직선
    if isfinite(res.t20) && isfinite(res.t80)
        y20 = interp1(t_ms, res.xf, res.t20);
        y80 = interp1(t_ms, res.xf, res.t80);
        plot([res.t20 res.t80], [y20 y80], 'LineWidth', 3);      % 20–80% 연결선
        plot(res.t20, y20, 'd', 'MarkerSize',8, 'LineWidth',1);  % 20%
        plot(res.t80, y80, 'd', 'MarkerSize',8, 'LineWidth',1);  % 80%
        % Amplitude(μ→peak) 보조선
        plot([res.peak_ms res.peak_ms], [res.mu res.xf(res.peak_idx)], '--', 'LineWidth', 1.25);
    end

    xlabel('Time (ms)'); ylabel('Signal (a.u.)');
    title(sprintf('fEPSP Features — Ch %d | Amp=%.3f | Slope(20–80%%)=%.4f | SNR=%.2f | %s', ...
        ch, res.amplitude, res.slope2080, res.snr, res.polarity));
    legend([p1 p2], {'Raw','Filtered'}, 'Location','best');

    % x축 범위 — [min,max]가 유효하지 않으면 자동
    tmin = min(t_ms,[],'omitnan'); tmax = max(t_ms,[],'omitnan');
    if ~isfinite(tmin) || ~isfinite(tmax) || tmin >= tmax
        xlim('auto');
    else
        xlim([tmin tmax]);
    end
end

% 모든 채널을 보시려면 NUM_SHOW = C; 로 변경하세요.