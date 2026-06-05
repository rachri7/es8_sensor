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
% Requires: RTC Toolbox installed and initialised (run rtc_init first)
% Units: time [ms], data [bits], rate [kbps]
% =========================================================================

clear; clc;

fprintf('=== Exercise 3: CAN Bus RTC Analysis ===\n\n');

% ---- Parameters ---------------------------------------------------------
L   = 100;   % packet length [bits]
T0  = 2;     % period of Node 0 [ms]
T1  = 5;     % period of Node 1 [ms]
T2  = 3;     % period of Node 2 [ms]
T_rx = 10;   % receive period for periodic reader [ms] (arbitrary)
C   = 200;   % CAN bus bitrate [kbps] -- arbitrary, must be > C_min

% =========================================================================
% PART 1: Minimum bitrate for all nodes to be schedulable
%
% Each node must be able to transmit at least one packet per period.
% The total offered load must not exceed the link capacity:
%   rho = (L/T0 + L/T1 + L/T2) / C  <=  1
%   => C_min = L*(1/T0 + 1/T1 + 1/T2)
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

% =========================================================================
% PART 2: Arrival curves for the three nodes
%
% Each node sends one packet of L bits every T_i ms (periodic, no jitter).
% rtcpjd(P, J, D) returns a single staircase arrival curve where:
%   P = period [ms], J = jitter = 0, D = min inter-arrival = 0
%
% These are event-level curves (packets, not bits) -- scaled to bits
% internally by rtcgpc via the execution demand ED = L bits/packet.
% =========================================================================

alpha0 = rtcpjd(T0, 0, 0);  % Node 0: one packet every T0 ms
alpha1 = rtcpjd(T1, 0, 0);  % Node 1: one packet every T1 ms
alpha2 = rtcpjd(T2, 0, 0);  % Node 2: one packet every T2 ms

% =========================================================================
% PART 3: Service curve of the CAN bus
%
% CAN bus modelled as a pure-rate server at C kbps (delay = 0).
% rtcbdu(0, C) gives the upper bounded-delay service curve: C*t bits
% served in any window of length t.
% =========================================================================

beta = rtcfs(C);   % C resource units per time unit = C kbps

% =========================================================================
% PART 4: Priority scheduling via Greedy Processing Component (GPC) chain
%
% Priority order (lower CAN-ID = higher priority) is determined by the
% order the GPC calls are chained -- first call gets full service, each
% subsequent call receives only what the previous left behind.
%   Node 0  <-- full service beta (highest priority, served first)
%   Node 1  <-- residual after Node 0
%   Node 2  <-- residual after Node 0 and Node 1
%
% Each rtcgpc call takes:
%   input:  arrival curve, available service curve, execution demand ED=L
%   output: a_out  -- output arrival curve after the bus
%           b_res  -- residual service curve for lower-priority nodes
%           del    -- worst-case delay from packet arriving to leaving bus
%           buf    -- worst-case buffer requirement [packets]
% =========================================================================

fprintf('--- Priority scheduling (GPC chain) ---\n');

% Node 0 (highest priority): gets the full service
[a0_out, b0_res, del0, buf0] = rtcgpc(alpha0, beta, L);
% del0 = 0.5ms = L/C = 100/200:
% must be EITHER own transmission time OR waiting for one blocking packet --
% cannot be both else result would be 1.0ms.
% both equal L/C = 0.5ms so result is the same either way.
fprintf('Node 0 (T0=%dms, highest priority, CAN-ID 0x000):\n', T0);
fprintf('  Max delay  = %.3f ms\n', del0);
fprintf('  Max buffer = %.0f pkts  (%d bits)\n\n', buf0, buf0*L);

% Node 1 (second priority): gets the residual after Node 0
[a1_out, b1_res, del1, buf1] = rtcgpc(alpha1, b0_res, L);
% del1 = 1.0ms = 2*L/C: one blocking packet + own transmission.
fprintf('Node 1 (T1=%dms, second priority,  CAN-ID 0x001):\n', T1);
fprintf('  Max delay  = %.3f ms\n', del1);
fprintf('  Max buffer = %.0f pkts  (%d bits)\n\n', buf1, buf1*L);

% Node 2 (lowest priority): gets the residual after Nodes 0 and 1
[a2_out, b2_res, del2, buf2] = rtcgpc(alpha2, b1_res, L);
% del2 = 1.5ms = 3*L/C: one blocking packet + Node 0 + own transmission.
fprintf('Node 2 (T2=%dms, lowest  priority, CAN-ID 0x002):\n', T2);
fprintf('  Max delay  = %.3f ms  <-- answer to exercise\n', del2);
fprintf('  Max buffer = %.0f pkts  (%d bits)\n\n', buf2, buf2*L);

% =========================================================================
% FIGURE 2: Input vs output arrival curves for all three nodes
%
% Solid   = input  alpha_i  (what the node sends onto the bus)
% Dashed  = output a_i_out  (what departs the bus after queuing)
%
% The output curve sits above the input -- the burst grows because
% bus delay allows packets to bunch closer together than they arrived.
% The higher the priority, the smaller the burst increase.
% =========================================================================

Max_Time_on_plot = 30;

figure(2); clf;
rtcplot(alpha0, 'r-',  ...   % Node 0 input
        a0_out, 'r--', ...   % Node 0 output
        alpha1, 'b-',  ...   % Node 1 input
        a1_out, 'b--', ...   % Node 1 output
        alpha2, 'm-',  ...   % Node 2 input
        a2_out, 'm--', ...   % Node 2 output
        Max_Time_on_plot, 'LineWidth', 1.5);

legend( sprintf('Node 0 input   \\alpha_0   (T_0=%d ms, highest prio)', T0), ...
        sprintf('Node 0 output  \\alpha_0^*'),                               ...
        sprintf('Node 1 input   \\alpha_1   (T_1=%d ms)',                T1), ...
        sprintf('Node 1 output  \\alpha_1^*'),                               ...
        sprintf('Node 2 input   \\alpha_2   (T_2=%d ms, lowest prio)',   T2), ...
        sprintf('Node 2 output  \\alpha_2^*'),                               ...
        'Location', 'northwest');
title('Input vs output arrival curves  (solid=input, dashed=output after bus)');
xlabel('Time t [ms]');
ylabel('Cumulative events [packets]');
grid on;

% =========================================================================
% FIGURE 3: Full service beta vs residual service curves
%
% beta   (black) -- full CAN service before any node is served
% b0_res (blue)  -- residual after Node 0: what Node 1 and Node 2 share
% b1_res (red)   -- residual after Node 1: what Node 2 receives
% b2_res (magenta)-- unused capacity after all three nodes
%
% Each residual is lower than the previous -- each node consumes its share.
% =========================================================================

figure(3); clf;
rtcplot(beta,   'k-', ...   % full service
        b0_res, 'b-', ...   % residual after Node 0
        b1_res, 'r-', ...   % residual after Node 1
        b2_res, 'm-', ...   % residual after Node 2
        Max_Time_on_plot, 'LineWidth', 1.5);
legend( sprintf('\\beta          full CAN service  C=%d kbps',         C),  ...
        sprintf('\\beta_0^{res}  residual after Node 0  (T_0=%d ms)', T0),  ...
        sprintf('\\beta_1^{res}  residual after Node 1  (T_1=%d ms)', T1),  ...
        sprintf('\\beta_2^{res}  residual after Node 2  (unused)'),         ...
        'Location', 'northwest');
title('Service curve and residuals after priority scheduling');
xlabel('Time t [ms]');
ylabel('Cumulative bits');
grid on;

% =========================================================================
% PART 5: Receive buffer for Node 2
%
% After the CAN bus, Node 2's packets arrive at a receive buffer.
% The buffer is drained by a PERIODIC READER with period T_rx [ms]:
%   - It reads exactly one packet every T_rx ms.
%   - Lower service curve: rtcpsl(S, P, B)
%       S = share allocated per period = 1 packet
%       P = period of the reader = T_rx  (must be integer)
%       B = bandwidth (unit: 1 packet/slot)
%
% Minimum buffer size (to avoid packet loss) =
%   maximum vertical distance between arrival and drain curves =
%   rtcv(a2_out, beta_rx_l)
% =========================================================================

fprintf('--- Receive buffer at Node 2 (T_rx = %d ms) ---\n', T_rx);

% Lower service curve of the periodic reader
beta_rx_l = rtcpsl(1, T_rx, 1);  % 1 packet per T_rx ms, unit bandwidth

% Minimum buffer size [events = packets]
% rtcv computes the maximum vertical distance (backlog) between arrival
% and drain curves -- requires both to be in the same event/packet domain.
buf_min = rtcv(a2_out, beta_rx_l);
fprintf('  Min buffer size = %.0f pkts  (%d bits)\n\n', buf_min, buf_min*L);

% =========================================================================
% FIGURE 4: Receive buffer -- Node 2 output vs periodic drain
%
% a2_out     (magenta) -- worst-case arrivals at the buffer.
%                         Slope = 1/T2 pkts/ms, burst larger than input
%                         because bus delay bunches packets together.
% beta_rx_l  (green)   -- periodic drain: 1 packet read every T_rx ms.
%                         Staircase -- flat between reads, steps up.
%
% The VERTICAL GAP between a2_out and beta_rx_l is the worst-case backlog.
% If a2_out slope > beta_rx_l slope (T_rx > T2), gap grows unbounded -> Inf.
% For finite buffer need T_rx < T2 = 3ms.
% =========================================================================

figure(4); clf;
rtcplot(a2_out,    'm-', ...
        beta_rx_l, 'g-', ...
        T_rx * 4, 'LineWidth', 1.5);

buf_min = rtcplotv(a2_out, beta_rx_l, 'g', 'LineWidth', 2);

legend( sprintf('\\alpha_2^*(t)  Node 2 output (worst-case arrivals)'),           ...
        sprintf('\\beta_{rx}(t)  periodic drain  T_{rx}=%d ms  (1 pkt/period)', T_rx), ...
        sprintf('rtcplotv        min buffer = %.0f pkts = %.0f bits', buf_min, buf_min*L), ...
        'Location', 'northwest');
title(sprintf('Receive buffer | delay = %.2f ms | Buffer = %.0f pkts (%d bits) | T_{rx}=%d ms', ...
              del2, buf_min, buf_min*L, T_rx));
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
    buf_test = rtcv(a2_out, b_rx_t);
    if isinf(buf_test)
        fprintf('%-12d  %-20s  %-20s\n', T_test, 'Inf', 'Inf (reader too slow)');
    else
        fprintf('%-12d  %-20.0f  %-20.0f\n', T_test, buf_test, buf_test*L);
    end
end
fprintf('\n');

fprintf('--- Parametric: Node 2 delay and buffer vs. bitrate C ---\n');
fprintf('  (T_rx = %d ms,  C sweeps above C_min = %.2f kbps)\n\n', T_rx, C_min);
fprintf('%-12s  %-15s  %-15s  %-15s\n', 'C [kbps]', 'rho', 'del2 [ms]', 'buf_min [pkts]');
fprintf('%s\n', repmat('-', 1, 60));

for C_test = [110 120 150 200 300 500 1000]
    % Rebuild GPC chain for this bitrate
    beta_t = rtcbdu(0, C_test);   % service curve at this bitrate

    [~, b0_t, ~, ~]    = rtcgpc(alpha0, beta_t, L);
    [~, b1_t, ~, ~]    = rtcgpc(alpha1, b0_t,   L);
    [a2_t, ~, d2_t, ~] = rtcgpc(alpha2, b1_t,   L);

    b_rx_t  = rtcpsl(1, T_rx, 1);
    buf_t   = rtcv(a2_t, b_rx_t);
    rho_t   = L*(1/T0 + 1/T1 + 1/T2) / C_test;

    if isinf(d2_t)
        fprintf('%-12d  %-15.3f  %-15s  %-15s\n', C_test, rho_t, 'Inf', 'Inf');
    else
        fprintf('%-12d  %-15.3f  %-15.3f  %-15.0f\n', C_test, rho_t, d2_t, buf_t);
    end
end