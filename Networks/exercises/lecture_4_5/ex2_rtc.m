% =========================================================================
% Exercise 2 - RTC Toolbox Analysis of Sensor Network (continuation of Ex 1)
% Networks and Systems, Lecture 4/5
%
% System: 8 sensors each producing 100 bits every 1ms, packetised into
% packets of N data samples + 100 bits overhead, sent over a shared 1Mbps
% link.  All 8 sensors share the same link -- aggregate arrival rate and
% burst depend on N.
% =========================================================================

clear; clc;

% ---- System parameters (from Exercise 1) --------------------------------
ns = 8;      % number of sensors
T  = 1;      % sampling period per sensor [ms]
O  = 100;    % fixed packet overhead (header + error detection) [bits]
s  = 1000;   % communication link speed [kbps]

fprintf('=== Exercise 2: RTC Toolbox Analysis ===\n\n');

% =========================================================================
% PART 1: Encode arrival and service curves for N = 4 (theoretical minimum)
% =========================================================================

% Compute Packet sizing
N     = 4;               % Data sample put in one packet
L     = 100*N + O;       % packet length [bits]: N data samples + overhead

% Compute IDK ...
P     = N * T;           % inter-packet period per sensor [ms]
r_s   = L / P;           % single-sensor long-term rate [kbps]
b_s   = L;               % single-sensor burst [bits]: one full packet

% Aggregated rate (r) and max backlog (b)
r_agg = ns * r_s;        % aggregate rate of all 8 sensors [kbps]
b_agg = ns * b_s;        % aggregate burst [bits]: 8 simultaneous packets

fprintf('--- N = %d ---\n', N);
fprintf('  Packet length  L     = %d bits\n',  L);
fprintf('  Inter-pkt period P   = %d ms\n',    P);
fprintf('  Aggregate rate r_agg = %d kbps\n',  r_agg);
fprintf('  Aggregate burst b_agg= %d bits\n\n',b_agg);

% =========================================================================
% PART 1: Modelled as Token Bucket (leaky bucket)
% =========================================================================


% Arrival curve: token bucket  alpha(t) = r_agg * t + b_agg
%   x0    = starting x-coordinate of the segment
%   y0    = starting y-coordinate of the segment (the burst intercept)
%   slope = rate of the segment (bits per ms = kbps)
alpha_agg = rtccurve([[0  b_agg  r_agg]]);

% Service curve: pure-rate link  beta(t) = s * t
beta = rtccurve([[0  0  s]]);

% =========================================================================
% PART 2: Worst-case delay (horizational) and backlog (vertical)
% =========================================================================

delay = rtch(alpha_agg, beta);
fprintf('Worst-case delay  (rtch): %.2f ms\n', delay);

backlog = rtcv(alpha_agg, beta);
fprintf('Worst-case backlog (rtcv): %.0f bits\n\n', backlog);

% =========================================================================
% PART 3 Build Figure -- all using RTC curve objects
% =========================================================================

X_MAX = 50;  % plot x-range [ms]


% backlog_curve = alpha - beta = bucket level over time:
% if stable (r_agg < s) the backlog drains to zero after the initial burst.
% rtcmax(c, 0)  clamps the curve so it never goes below 0.
backlog_curve = rtcmax(rtcminus(alpha_agg, beta), 0);

figure(1); clf;

% (1) Plot the three main RTC curves in one rtcplot call
rtcplot(beta,          'r--', ...    % service:  beta(t)   = s*t
        alpha_agg,     'b--',  ...   % arrival:  alpha(t)  = r_agg*t + b_agg
        backlog_curve, 'm--',  ...   % backlog:  alpha - beta (evolution over t)
        X_MAX, 'LineWidth', 1.5);

% (2) Toolbox markers - capture scalar values as return arguments
%     rtcplotv draws a line where y=backlog_max
%     rtcploth draws a line where x=delay_max
backlog_max = rtcplotv(alpha_agg, beta, 'g', 'LineWidth', 2);
delay_max   = rtcploth(alpha_agg, beta, 'c', 'LineWidth', 2);


% (4) Legend with captured scalar values
legend(sprintf('\\beta(t) = %dt  (service)', s), ...
       sprintf('\\alpha(t) = %dt + %d  (arrival, N=%d)', r_agg, b_agg, N), ...
       sprintf('\\alpha(t) - \\beta(t)  (backlog evolution)'), ...
       sprintf('rtcplotv: max backlog = %.0f bits', backlog_max), ...
       sprintf('rtcploth: max delay   = %.1f ms',   delay_max), ...
       'Location', 'northwest');
title(sprintf('N=%d  |  Max backlog = %.0f bits  |  Max delay = %.1f ms', ...
              N, backlog_max, delay_max));
xlabel('Time t [ms]'); ylabel('Cumulative bits');
grid on;

% =========================================================================
% PART 5: Find the optimal N using the RTC toolbox
% Sweep N=1..8: the minimum N where rtch is FINITE is the optimal value.
% =========================================================================

fprintf('--- Optimal N sweep (using rtch and rtcv) ---\n');
fprintf('%-4s  %-12s  %-6s  %-12s  %-12s\n', ...
    'N', 'r_agg[kbps]', 'rho', 'delay[ms]', 'backlog[b]');
fprintf('%s\n', repmat('-', 1, 55));
for N_val = 1:8
    L_n   = 100*N_val + O;              % Packet length
    r_n   = ns * L_n / (N_val*T);       % aggregate rate of all 8 sensors [kbps] 
    b_n   = ns * L_n;                   % aggregate burst [bits]: 8 simultaneous packets
    rho_n = r_n / s;                    % Utilization or what 
    a_n = rtccurve([[0  b_n  r_n]]);    % create arrival curve given n_val data points
    d_n = rtch(a_n, beta);              % Max Delay 
    v_n = rtcv(a_n, beta);              % Max Backlog 
    if isinf(d_n)
        fprintf('%-4d  %-12.1f  %-6.3f  %-12s  %-12s\n', ...
                N_val, r_n, rho_n, 'Inf', 'Inf');
    else
        fprintf('%-4d  %-12.1f  %-6.3f  %-12.2f  %-12.0f\n', ...
                N_val, r_n, rho_n, d_n, v_n);
    end
end

fprintf('\nConclusion: optimal N = 4 (min finite delay = %.1f ms)\n', 4000/s);
fprintf('Practical:  use  N = 5 for 40 kbps spare capacity.\n');
