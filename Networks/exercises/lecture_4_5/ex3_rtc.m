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
% Upper PJD arrival curve: rtcpjdu(P, J, D) counts packets in any window.
%   P = period [ms] (integer), J = jitter = 0, D = min inter-arrival = 0
% Lower PJD: rtcpjdl(P) is the matching lower bound.
%
% These are EVENT-LEVEL curves (counting packets, not bits).
% The execution demand ED = L bits/packet scales them to bit-level service.
% =========================================================================

alpha0_u = rtcpjdu(T0);  % Node 0 upper: at most ceil((t+0)/T0) pkts in [0,t]
alpha0_l = rtcpjdl(T0);  % Node 0 lower: at least floor(t/T0) pkts
alpha1_u = rtcpjdu(T1);  % Node 1 upper
alpha1_l = rtcpjdl(T1);  % Node 1 lower
alpha2_u = rtcpjdu(T2);  % Node 2 upper (lowest priority)
alpha2_l = rtcpjdl(T2);  % Node 2 lower

% =========================================================================
% PART 3: Service curve of the CAN bus
%
% The CAN bus provides a continuous service at rate C kbps.
% We model it as a bounded-delay resource with:
%   delay = 0  (no artificial delay -- just raw bit rate)
%   bandwidth = C [kbps]
%
% rtcbdu(D, B) / rtcbdl(D, B) : upper/lower bounded-delay service curves.
% With D=0: beta(t) = B*t  (pure rate).
% =========================================================================

beta_u = rtcbdu(0, C);   % upper service curve for CAN bus
beta_l = rtcbdl(0, C);   % lower service curve for CAN bus

% =========================================================================
% PART 4: Priority scheduling via Greedy Processing Component (GPC) chain
%
% In RTC, strict-priority scheduling on a shared resource is modelled as a
% chain of GPCs.  The highest-priority flow is processed first and consumes
% service from the full resource.  Each subsequent flow receives only the
% RESIDUAL service left after higher-priority traffic has been served.
%
% rtcgpc(AU, AL, BU, BL, ED) -- Greedy Processing Component:
%   AU, AL = upper/lower arrival curves of the flow [events]
%   BU, BL = upper/lower service curves available to this flow [bits]
%   ED     = execution demand per event [bits/packet] = L
%
%   Returns:
%     [AU_out, AL_out] = output arrival curves of the processed flow
%     [BU_res, BL_res] = RESIDUAL service curves (what is left for lower-prio flows)
%     del              = worst-case delay for this flow [ms]
%     buf              = worst-case buffer requirement [events]
%
% Chain:
%   Node 0  <-- full service  beta
%   Node 1  <-- residual after Node 0
%   Node 2  <-- residual after Nodes 0 and 1
% =========================================================================

fprintf('--- Priority scheduling (GPC chain) ---\n');

% Node 0 (highest priority): gets the full service
[a0u_out, a0l_out, b0u_res, b0l_res, del0, buf0] = ...
    rtcgpc(alpha0_u, alpha0_l, beta_u, beta_l, L);

fprintf('Node 0 (T0=%dms, highest priority, CAN-ID 0x000):\n', T0);
fprintf('  Max p2p delay  = %.3f ms\n', del0);
fprintf('  Max buffer     = %.0f pkts  (%d bits)\n\n', buf0, buf0*L);

% Node 1 (second priority): gets the residual after Node 0
[a1u_out, a1l_out, b1u_res, b1l_res, del1, buf1] = ...
    rtcgpc(alpha1_u, alpha1_l, b0u_res, b0l_res, L);

fprintf('Node 1 (T1=%dms, second priority,  CAN-ID 0x001):\n', T1);
fprintf('  Max p2p delay  = %.3f ms\n', del1);
fprintf('  Max buffer     = %.0f pkts  (%d bits)\n\n', buf1, buf1*L);

% Node 2 (lowest priority): gets the residual after Nodes 0 and 1
[a2u_out, a2l_out, b2u_res, b2l_res, del2, buf2] = ...
    rtcgpc(alpha2_u, alpha2_l, b1u_res, b1l_res, L);

fprintf('Node 2 (T2=%dms, lowest  priority, CAN-ID 0x002):\n', T2);
fprintf('  Max p2p delay  = %.3f ms  <-- answer to exercise\n', del2);
fprintf('  Max buffer     = %.0f pkts  (%d bits)\n\n', buf2, buf2*L);

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
%   rtcv(a2u_out, beta_rx_l)
% =========================================================================

fprintf('--- Receive buffer at Node 2 (T_rx = %d ms) ---\n', T_rx);

% Lower service curve of the periodic reader
beta_rx_l = rtcpsl(1, T_rx, 1);  % 1 packet per T_rx ms, unit bandwidth

% Minimum buffer size [events = packets]
buf_min = rtcv(a2u_out, beta_rx_l);
%  rtcplotv draws a vertical spike at x=0 from y=0 to y=backlog_max 
% (min buffer needed )
fprintf('  Min buffer size = %.0f pkts  (%d bits)\n\n', buf_min, buf_min*L);

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

fprintf('--- Parametric: Node 2 p2p delay and buffer vs. bitrate C ---\n');
fprintf('  (T_rx = %d ms,  C sweeps above C_min = %.2f kbps)\n\n', T_rx, C_min);
fprintf('%-12s  %-15s  %-15s  %-15s\n', 'C [kbps]', 'rho', 'del2 [ms]', 'buf_min [pkts]');
fprintf('%s\n', repmat('-', 1, 60));

for C_test = [110 120 150 200 300 500 1000]
    % Rebuild GPC chain for this bitrate
    bu_t = rtcbdu(0, C_test);
    bl_t = rtcbdl(0, C_test);

    [~, ~, b0u_t, b0l_t, ~, ~] = rtcgpc(alpha0_u, alpha0_l, bu_t,   bl_t,   L);
    [~, ~, b1u_t, b1l_t, ~, ~] = rtcgpc(alpha1_u, alpha1_l, b0u_t,  b0l_t,  L);
    [a2u_t, ~, ~, ~, d2_t, ~]  = rtcgpc(alpha2_u, alpha2_l, b1u_t,  b1l_t,  L);

    b_rx_t  = rtcpsl(1, T_rx, 1);
    buf_t   = rtcv(a2u_t, b_rx_t);
    rho_t   = L*(1/T0 + 1/T1 + 1/T2) / C_test;

    if isinf(d2_t)
        fprintf('%-12d  %-15.3f  %-15s  %-15s\n', C_test, rho_t, 'Inf', 'Inf');
    else
        fprintf('%-12d  %-15.3f  %-15.3f  %-15.0f\n', C_test, rho_t, d2_t, buf_t);
    end
end
