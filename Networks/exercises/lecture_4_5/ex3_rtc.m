% =========================================================================
% Exercise 3 - CAN Bus RTC Analysis with Priority Scheduling
% Networks and Systems, Lecture 4/5
%
% Topology: 3 nodes on a single CAN bus segment.
% Priority order (lower CAN ID = higher priority):
%   Node 0 : period T0=2ms,  CAN-ID 0x000  (HIGHEST priority)
%   Node 1 : period T1=5ms,  CAN-ID 0x001  (second  priority)
%   Node 2 : period T2=3ms,  CAN-ID 0x002  (LOWEST  priority)
%
% All packets: L=100 bits.  CAN bus bitrate C [kbps] (arbitrary > C_min).
% Receive buffer at Node 2, drained by periodic reader with period T_rx [ms].
%
% =========================================================================

clear; clc;

fprintf('=== Exercise 3: CAN Bus RTC Analysis ===\n\n');

% ---- Parameters ---------------------------------------------------------
L   = 100;   % packet length [bits]
T0  = 2;     % period of Node 0 [ms]
T1  = 5;     % period of Node 1 [ms]
T2  = 3;     % period of Node 2 [ms]
T_rx = 2;   % receive period for periodic reader [ms] (arbitrary)
C   = 200;   % CAN bus bitrate [kbps] -- arbitrary, must be > C_min

% =========================================================================
% PART 1: Minimum bitrate for all nodes to be schedulable
% =========================================================================

C_min = L * (1/T0 + 1/T1 + 1/T2);
fprintf('--- Minimum bitrate ---\n');
fprintf('  C_min = L*(1/T0 + 1/T1 + 1/T2)\n');
fprintf('        = %d*(1/%d + 1/%d + 1/%d)\n', L, T0, T1, T2);
fprintf('        = %.4f kbps\n', C_min);
fprintf('  --> Choose any C > %.2f kbps.  Using C = %d kbps.\n\n', C_min, C);
if C <= C_min
    error('C = %.1f kbps is not above C_min = %.2f kbps!', C, C_min);
end

% Arrival curves in EVENTS (packets) -- rtcgpc scales by ED=L internally.
% Modelled as token bucket  alpha(t) = b + r*t
%   Upper (b=1, r=1/T): worst-case -- each node fires at most 1 pkt instantly
%   Lower (b=0, r=1/T): best-case  -- no burst, arrivals perfectly spread
alpha0_u = rtccurve([[0, 1, 1/T0]]);  % Node 0 upper: burst=1, rate=1/T0
alpha0_l = rtccurve([[0, 0, 1/T0]]);  % Node 0 lower: burst=0, rate=1/T0
alpha1_u = rtccurve([[0, 1, 1/T1]]);  % Node 1 upper
alpha1_l = rtccurve([[0, 0, 1/T1]]);  % Node 1 lower
alpha2_u = rtccurve([[0, 1, 1/T2]]);  % Node 2 upper
alpha2_l = rtccurve([[0, 0, 1/T2]]);  % Node 2 lower

% =========================================================================
% PART 3: Service curve of the CAN bus
% CAN bus modelled as a pure-rate server at C kbps (delay = 0).
% =========================================================================
beta_u = rtcbdu(0, C);   % upper service curve for CAN bus
beta_l = rtcbdl(0, C);   % lower service curve for CAN bus

% =========================================================================
% PART 4: Priority scheduling via Greedy Processing Component (GPC) chain
%
% Priority order (lower CAN-ID = higher priority) is determined by the
% order the GPC calls are chained -- first call gets full service, each
% subsequent call receives only what the previous left behind.
%   Node 0  <-- full service beta (highest priority, served first)
%   Node 1  <-- residual after Node 0
%   Node 2  <-- residual after Node 0 and Node 1
% =========================================================================
% (L/C = 100/200 = 0.5 ms per packet)


fprintf('--- Priority scheduling (GPC chain) ---\n');

% Node 0 (highest priority): gets the full service
[a0u_out, a0l_out, b0u_res, b0l_res, del0, buf0] = ...
    rtcgpc(alpha0_u, alpha0_l, beta_u, beta_l, L);

fprintf('Node 0 (T0=%dms, highest priority, CAN-ID 0x000):\n', T0);
fprintf('  Max p2p delay  = %.3f ms\n', del0);
fprintf('  Max buffer     = %d pkts  (%d bits)\n\n', ceil(buf0), ceil(buf0)*L);

% Node 1 (second priority): gets the residual after Node 0
[a1u_out, a1l_out, b1u_res, b1l_res, del1, buf1] = ...
    rtcgpc(alpha1_u, alpha1_l, b0u_res, b0l_res, L);

fprintf('Node 1 (T1=%dms, second priority,  CAN-ID 0x001):\n', T1);
fprintf('  Max p2p delay  = %.3f ms\n', del1);
fprintf('  Max buffer     = %d pkts  (%d bits)\n\n', ceil(buf1), ceil(buf1)*L);

% Node 2 (lowest priority): gets the residual after Nodes 0 and 1
[a2u_out, a2l_out, b2u_res, b2l_res, del2, buf2] = ...
    rtcgpc(alpha2_u, alpha2_l, b1u_res, b1l_res, L);

fprintf('Node 2 (T2=%dms, lowest  priority, CAN-ID 0x002):\n', T2);
fprintf('  Max p2p delay  = %.3f ms  <-- answer to exercise\n', del2);
fprintf('  Max buffer     = %d pkts  (%d bits)\n\n', ceil(buf2), ceil(buf2)*L);

% ========================= RESULTS ========================================
% GPC  (full) = access delay + own transmission time (L/C_res).
% Later Figures only shows access delay
%
%   Node 0: 1.000ms - 0.500ms = 0.500ms = L/C      = 100/200  (own tx time)
%           Fig5 shows 0.500ms = access delay  |  T0=2ms, L/T0=50 kbps, C_res=200 kbps
%
%   Node 1: 2.000ms - 1.333ms = 0.667ms = L/C_res1 = 100/150  (own tx time)
%           Fig5 shows 1.333ms = access delay  |  T1=5ms, L/T1=20 kbps, C_res=150 kbps
%
%   Node 2: 3.077ms - 2.308ms = 0.769ms = L/C_res2 = 100/130  (own tx time)
%           Fig5 shows 2.308ms = access delay  |  T2=3ms, L/T2=33 kbps, C_res=130 kbps
%==========================================================================
 
% =========================================================================
% FIGURE 1: Per-node delay (h) and buffer (v) -- input upper vs service lower
% NOTE: Fig5 (rtch) = ACCESS delay (time waiting to get onto CAN bus).
% NOT P2P Delay
% =========================================================================
Max_zoom = 5;

% need to make new becuase new to scale to L (before gpc did this)
a0u_b = rtccurve([[0, L, L/T0]]);  % Node 0 upper in bits
a1u_b = rtccurve([[0, L, L/T1]]);  % Node 1 upper in bits
a2u_b = rtccurve([[0, L, L/T2]]);  % Node 2 upper in bits

% Pre-compute delay [ms] and buffer [bits] so values go into legend labels
d0 = rtch(a0u_b, beta_l);   v0 = rtcv(a0u_b, beta_l);
d1 = rtch(a1u_b, b0l_res);  v1 = rtcv(a1u_b, b0l_res);
d2 = rtch(a2u_b, b1l_res);  v2 = rtcv(a2u_b, b1l_res);

figure(5); clf;
rtcplot(a0u_b,   'r-',  ...   % Node 0 arrival upper
        beta_l,  'r--', ...   % Node 0 service: full CAN lower
        a1u_b,   'b-',  ...   % Node 1 arrival upper
        b0l_res, 'b--', ...   % Node 1 service: residual after Node 0
        a2u_b,   'g-',  ...   % Node 2 arrival upper
        b1l_res, 'g--', ...   % Node 2 service: residual after Nodes 0+1
        Max_zoom, 'LineWidth', 1.5);

% Draw h (delay) and v (buffer) markers for each node
rtcploth(a0u_b, beta_l,  'r', 'LineWidth', 2);  rtcplotv(a0u_b, beta_l,  'r', 'LineWidth', 2);
rtcploth(a1u_b, b0l_res, 'b', 'LineWidth', 2);  rtcplotv(a1u_b, b0l_res, 'b', 'LineWidth', 2);
rtcploth(a2u_b, b1l_res, 'g', 'LineWidth', 2);  rtcplotv(a2u_b, b1l_res, 'g', 'LineWidth', 2);

legend( sprintf('Node 0  \\alpha upper  T_0=%dms  -->  delay=%.3fms  buf=%dpkts', T0, d0, ceil(v0/L)), ...
        sprintf('Node 0  \\beta lower  (full CAN)'),                                                    ...
        sprintf('Node 1  \\alpha upper  T_1=%dms  -->  delay=%.3fms  buf=%dpkts', T1, d1, ceil(v1/L)), ...
        sprintf('Node 1  \\beta_0^{res} lower'),                                                        ...
        sprintf('Node 2  \\alpha upper  T_2=%dms  -->  delay=%.3fms  buf=%dpkts', T2, d2, ceil(v2/L)), ...
        sprintf('Node 2  \\beta_1^{res} lower'),                                                        ...
        'Location', 'northwest');
title('Per-node worst-case delay (horizontal) and buffer (vertical)');
xlabel('Time t [ms]');
ylabel('Cumulative bits');
grid on;

% =========================================================================
% PART 5: Receive buffer for Node 2
% =========================================================================
Max_Time_on_plot = 30;

fprintf('--- Receive buffer at Node 2 (T_rx = %d ms) ---\n', T_rx);

% Lower service curve of the periodic reader
beta_rx_l = rtcpsl(1, T_rx, 1);  % 1 packet per T_rx ms, unit bandwidth

% Minimum buffer size [events = packets]
buf_min = rtcv(a2u_out, beta_rx_l);
%  rtcplotv draws a vertical spike at x=0 from y=0 to y=backlog_max 
% (min buffer needed )
fprintf('  Min buffer size = %.0f pkts  (%d bits)\n\n', buf_min, buf_min*L);
 
figure(4); clf;
rtcplot(a2u_out,   'm-', ...   % Node 2 upper arrival (worst-case)
        beta_rx_l, 'g--', ...   % receive drain lower  (worst-case)
        Max_Time_on_plot, 'LineWidth', 1.5);

buf_min = rtcplotv(a2u_out, beta_rx_l, 'g', 'LineWidth', 2);

legend( sprintf('\\alpha_2^*(t)  Node 2 upper arrival  (worst-case)'),              ...
        sprintf('\\beta_{rx}(t)  periodic drain  T_{rx}=%d ms', T_rx),              ...
        sprintf('min buffer = %.0f pkts = %.0f bits', buf_min, buf_min*L),           ...
        'Location', 'northwest');
title(sprintf('Receive buffer | Buffer = %.0f pkts (%d bits) | T_{rx}=%d ms', ...
               buf_min, buf_min*L, T_rx));
xlabel('Time t [ms]');
ylabel('Cumulative events [packets]');
grid on;


% =========================================================================
% PART 6: Parametric analysis -- vary T_rx and C
%
% The exercise asks for results for "arbitrary" receive period and bitrate.
% We sweep both to show how the results depend on the parameters.
% =========================================================================

fprintf('--- Parametric: min buffer vs. receive period T_rx (C = %d kbps) ---\n', C);
fprintf('%-12s  %-20s  %-20s\n', 'T_rx [ms]', 'Buffer [pkts]', 'Buffer [bits]');
fprintf('%s\n', repmat('-', 1, 55));

for T_test = [2 3 4 5 6 8 10 15 20]
    if T_test < 1, continue; end
    b_rx_t   = rtcpsl(1, T_test, 1);
    buf_test = rtcv(a2u_out, b_rx_t);
    if isinf(buf_test)
        fprintf('%-12d  %-20s  %-20s\n', T_test, 'Inf', 'Inf (reader too slow)');
    else
        fprintf('%-12d  %-20.0f  %-20.0f\n', T_test, buf_test, buf_test*L);
    end
end
fprintf('\n');

% NOTE: T_rx must be < T2 = 3ms, otherwise drain rate < Node 2 arrival rate -> Inf buffer.
% Use T_rx = 2ms for C sweep so buffer is always finite.
T_rx_csweep = 2;
fprintf('--- Parametric: Node 2 p2p delay and buffer vs. bitrate C ---\n');
fprintf('  (T_rx = %d ms,  C sweeps above C_min = %.2f kbps)\n\n', T_rx_csweep, C_min);
fprintf('%-12s  %-15s  %-15s  %-15s\n', 'C [kbps]', 'rho', 'del2 [ms]', 'buf_min [pkts]');
fprintf('%s\n', repmat('-', 1, 60));

for C_test = [110 120 150 200 300 500 1000]
    % Rebuild GPC chain for this bitrate
    bu_t = rtcbdu(0, C_test);
    bl_t = rtcbdl(0, C_test);

    [~, ~, b0u_t, b0l_t, ~, ~] = rtcgpc(alpha0_u, alpha0_l, bu_t,   bl_t,   L);
    [~, ~, b1u_t, b1l_t, ~, ~] = rtcgpc(alpha1_u, alpha1_l, b0u_t,  b0l_t,  L);
    [a2u_t, ~, ~, ~, d2_t, ~]  = rtcgpc(alpha2_u, alpha2_l, b1u_t,  b1l_t,  L);

    b_rx_t  = rtcpsl(1, T_rx_csweep, 1);
    buf_t   = rtcv(a2u_t, b_rx_t);
    rho_t   = L*(1/T0 + 1/T1 + 1/T2) / C_test;

    if isinf(d2_t)
        fprintf('%-12d  %-15.3f  %-15s  %-15s\n', C_test, rho_t, 'Inf', 'Inf');
    else
        fprintf('%-12d  %-15.3f  %-15.3f  %-15.0f\n', C_test, rho_t, d2_t, buf_t);
    end
end
