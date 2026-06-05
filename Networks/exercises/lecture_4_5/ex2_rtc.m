% =========================================================================
% Exercise 2 - RTC Toolbox Analysis of Sensor Network (continuation of Ex 1)
% Networks and Systems, Lecture 4/5
%
% Requires: RTC Toolbox installed and initialised (run rtc_init first)
% Units throughout: time in [ms], data in [bits], rate in [kbps = bits/ms]
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

N     = 4;
L     = 100*N + O;       % packet length [bits]: N data samples + overhead
P     = N * T;           % inter-packet period per sensor [ms]
r_s   = L / P;           % single-sensor long-term rate [kbps]
b_s   = L;               % single-sensor burst [bits]: one full packet
r_agg = ns * r_s;        % aggregate rate of all 8 sensors [kbps]
b_agg = ns * b_s;        % aggregate burst [bits]: 8 simultaneous packets

fprintf('--- N = %d ---\n', N);
fprintf('  Packet length  L     = %d bits\n',  L);
fprintf('  Inter-pkt period P   = %d ms\n',    P);
fprintf('  Aggregate rate r_agg = %d kbps\n',  r_agg);
fprintf('  Aggregate burst b_agg= %d bits\n\n',b_agg);

% Arrival curve: token bucket  alpha(t) = r_agg * t + b_agg
% rtccurve([[x0  y0  slope]]) defines a piecewise-linear curve where:
%   x0    = starting x-coordinate of the segment
%   y0    = starting y-coordinate of the segment (the burst intercept)
%   slope = rate of the segment (bits per ms = kbps)
alpha_agg = rtccurve([[0  b_agg  r_agg]]);

% Service curve: pure-rate link  beta(t) = s * t
% rtcbdl(D, B) creates the lower bounded-delay service curve with:
%   D = maximum delay guaranteed (0 here: no extra latency, just raw rate)
%   B = link bandwidth [kbps]
% Shape: beta(t) = 0 for t in [0, D], then rises with slope B.
% With D=0 this collapses to beta(t) = B*t.
beta = rtcbdl(0, s);

% =========================================================================
% PART 2: Worst-case delay via rtch (maximum Horizontal distance)
%
% rtch(alpha, beta) returns the maximum horizontal distance between the
% two curves: the maximum right-ward shift d such that alpha(t) <= beta(t+d).
% This is the worst-case queuing delay any bit can experience.
%
% For alpha(t) = r*t + b  and  beta(t) = s*t:
%   r > s  -->  delay = Inf  (arrival rate exceeds link rate -- UNSTABLE)
%   r = s  -->  delay = b/s  (parallel curves, constant gap forever)
%   r < s  -->  delay = b/s  (worst case at t=0, then decreases)
% =========================================================================

delay = rtch(alpha_agg, beta);
fprintf('Worst-case delay  (rtch): %.2f ms\n', delay);

% =========================================================================
% PART 3: Worst-case backlog via rtcv (maximum Vertical distance)
%
% rtcv(alpha, beta) returns sup_t { alpha(t) - beta(t) }:
% the maximum number of bits queued at the link at any instant.
%
%   r > s  -->  Inf (queue grows without bound)
%   r = s  -->  b_agg (constant permanent backlog, never drains)
%   r < s  -->  b_agg (worst case at t=0, then decreases to 0)
% =========================================================================

backlog = rtcv(alpha_agg, beta);
fprintf('Worst-case backlog (rtcv): %.0f bits\n\n', backlog);

% =========================================================================
% PART 4: Understanding what rtcploth and rtcplotv actually draw
%
% IMPORTANT: these functions do NOT plot the delay/backlog over time.
% They each draw a single LINE SEGMENT marking the GEOMETRIC LOCATION of
% the maximum deviation on the curve diagram (x = time, y = bits):
%
%   rtcplotv --> draws a VERTICAL line at t* showing max vertical gap
%                (t* = 0 for our token bucket, gap = b_agg bits)
%   rtcploth --> draws a HORIZONTAL line at y* showing max horizontal gap
%                (y* = b_agg bits, spanning d_max = b_agg/s ms in time)
%
% To actually SEE the evolution of backlog and delay over time, we need
% to plot alpha(t) and beta(t) on the same axes (Part 5 below).
% =========================================================================

% =========================================================================
% PART 4 (cont.): Build Figure 1 -- all using RTC curve objects
%
% rtcplot syntax:  rtcplot(C1,'style1', C2,'style2', ..., x_max, 'prop',val)
%   - Curve/style pairs come first, each with its own style string
%   - The optional x_max scalar comes after all curve/style pairs
%   - Any extra name/value line properties (LineWidth etc.) come last
%
% rtcplotv / rtcploth return the scalar max value when called with an
% output argument:  val = rtcplotv(alpha, beta)
% =========================================================================

X_MAX = 50;  % plot x-range [ms]

% --- Backlog evolution curve: alpha(t) - beta(t), clamped to >= 0 ---
% rtcminus(a,b) returns a Curve object for a-b.
% rtcmax(c, 0)  clamps the curve so it never goes below 0.
backlog_curve = rtcmax(rtcminus(alpha_agg, beta), 0);

figure(1); clf;

% (1) Plot the three main RTC curves in one rtcplot call
rtcplot(beta,          'r--', ...   % service:  beta(t)   = s*t
        alpha_agg,     'b--',  ...   % arrival:  alpha(t)  = r_agg*t + b_agg
        backlog_curve, 'm--',  ...   % backlog:  alpha - beta (evolution over t)
        X_MAX, 'LineWidth', 1.5);

% (2) Toolbox markers -- capture scalar values as return arguments
%     rtcplotv draws a vertical spike at x=0 from y=0 to y=backlog_max
%     rtcploth draws a horizontal line at y=b_agg from x=0 to x=delay_max
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
%
% Sweep N=1..8: the minimum N where rtch is FINITE is the optimal value.
% =========================================================================

fprintf('--- Optimal N sweep (using rtch and rtcv) ---\n');
fprintf('%-4s  %-12s  %-6s  %-12s  %-12s\n', ...
    'N', 'r_agg[kbps]', 'rho', 'delay[ms]', 'backlog[b]');
fprintf('%s\n', repmat('-', 1, 55));
for N_val = 1:8
    L_n   = 100*N_val + O;
    r_n   = ns * L_n / (N_val*T);
    b_n   = ns * L_n;
    rho_n = r_n / s;
    a_n = rtccurve([[0  b_n  r_n]]);
    d_n = rtch(a_n, beta);
    v_n = rtcv(a_n, beta);
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
