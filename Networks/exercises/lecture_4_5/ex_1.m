% =========================================================================
% Exercise 1 & 2 - Sensor Network DNC Analysis
% Networks and Systems, Lecture 4/5
%
% Exercise 1: Analytical derivation of arrival curves and optimal N
%
% Units throughout: time in [ms], data in [bits], rate in [kbps = bits/ms]
% =========================================================================
 
clear; clc;
 
% ---- System parameters --------------------------------------------------
ns = 8;      % number of sensors
T  = 1;      % sampling period per sensor [ms]
O  = 100;    % fixed packet overhead (header + error detection) [bits]
s  = 1000;   % communication link speed [kbps = bits/ms]  (1 Mbps)
 
% =========================================================================
% EXERCISE 1: Analytical DNC Derivation
% =========================================================================
 
fprintf('=== Exercise 1: Analytical DNC Derivation ===\n\n');
 
% -------------------------------------------------------------------------
% PART 1: Packet structure and per-sensor arrival curve
%
% Each sensor produces 100 bits every T=1 ms.
% After collecting N data samples, a single packet is sent containing:
%   - N * 100 bits of measurement data
%   - O = 100 bits of overhead (header, error detection, ...)
%
% So the packet length is:
%   L(N) = 100*N + O  [bits]
%
% The packet is sent once every N*T ms (one packet per N sampling periods):
%   P(N) = N * T      [ms]  -- inter-packet period
%
% The per-sensor arrival curve is a token-bucket  alpha_s(t) = r_s*t + b_s
% where:
%   b_s = L(N)          [bits]  -- burst: one full packet can arrive at once
%   r_s = L(N) / P(N)   [kbps]  -- long-run average rate
% -------------------------------------------------------------------------
 
% Each sensor produces 100 bits every T=1 ms.
% After collecting N data samples, a single packet is sent containing:

fprintf('--- Part 1: Per-sensor arrival curve (general N) ---\n');
fprintf('  L(N)   = 100*N + %d  [bits]\n', O);                             
fprintf('  P(N)   = N * %d      [ms]\n',   T);
fprintf('  r_s(N) = L(N)/P(N)  = (100*N + %d) / (N*%d)  [kbps]\n', O, T);
fprintf('  b_s(N) = L(N)       = 100*N + %d              [bits]\n\n', O);
 
% -------------------------------------------------------------------------
% PART 2: Aggregate arrival curve (all 8 sensors, independent flows)
%
% Assuming all 8 sensors share the same N, the aggregate token-bucket is:
%   r_agg(N) = ns * r_s(N)   [kbps]
%   b_agg(N) = ns * b_s(N)   [bits]
%
% This uses the token-bucket aggregation property:
%   sum of (r_i*t + b_i)  =  (sum r_i)*t + (sum b_i)
% which holds because the sensors are independent and share the same N.
% -------------------------------------------------------------------------
 
fprintf('--- Part 2: Aggregate arrival curve (all %d sensors) ---\n', ns);
fprintf('  r_agg(N) = %d * r_s(N)  =  %d*(100*N + %d) / (N*%d)  [kbps]\n', ns, ns, O, T);
fprintf('  b_agg(N) = %d * b_s(N)  =  %d*(100*N + %d)           [bits]\n\n', ns, ns, O);
 
% -------------------------------------------------------------------------
% PART 3: Maximum link utilisation and stability condition
%
% The link is a pure-rate service curve:  beta(t) = s * t
%
% For the system to be stable (finite delay and backlog), the aggregate
% long-run arrival rate must NOT exceed the link speed:
%   r_agg(N) <= s
%
%   ns * (100*N + O) / (N*T) <= s
%   ns * (100 + O/N)         <= s          [divide both sides by 1]
%   800 + 800/N              <= 1000
%   800/N                    <= 200
%   N                        >= 800/200 = 4
%
% Therefore the MINIMUM value of N for stability is N_min = 4.
%
% The link utilisation (rho) for a given N is:
%   rho(N) = r_agg(N) / s
% -------------------------------------------------------------------------
 
fprintf('--- Part 3: Stability condition and maximum utilisation ---\n');
fprintf('  Stability requires:  r_agg(N) <= s\n');
fprintf('  => ns*(100*N + O) / (N*T) <= s\n');
fprintf('  => %d*(100 + %d/N)       <= %d\n', ns, O, s);
fprintf('  => 800 + 800/N           <= 1000\n');
fprintf('  => N >= 800/200 = 4\n\n');
fprintf('  Minimum stable N:  N_min = 4\n\n');
 
fprintf('  Utilisation rho(N) = r_agg(N) / s:\n');
fprintf('  %-4s  %-14s  %-10s  %-12s  %-12s\n', ...
        'N', 'L [bits]', 'P [ms]', 'r_agg [kbps]', 'rho');
fprintf('  %s\n', repmat('-', 1, 58));
for N_val = 1:8
    L_n   = 100*N_val + O;
    P_n   = N_val * T;
    r_n   = ns * L_n / P_n;
    rho_n = r_n / s;
    stable = '';
    if rho_n <= 1, stable = '<= 1  STABLE'; else, stable = '> 1   UNSTABLE'; end
    fprintf('  %-4d  %-14d  %-10d  %-12.1f  %-6.3f  %s\n', ...
            N_val, L_n, P_n, r_n, rho_n, stable);
end
 
% -------------------------------------------------------------------------
% PART 4: Worst-case delay and backlog (analytical, for N = N_min = 4)
%
% For a token-bucket arrival  alpha(t) = r*t + b  served by a pure-rate
% link  beta(t) = s*t  with r <= s, the DNC bounds are:
%
%   Worst-case delay   d* = b_agg / s          [ms]
%   Worst-case backlog q* = b_agg              [bits]
%
% These are tight: the maximum horizontal distance (delay) between alpha
% and beta occurs at t=0 (the burst), giving d* = b_agg / s.
% The maximum vertical distance (backlog) is also b_agg at t=0.
% -------------------------------------------------------------------------