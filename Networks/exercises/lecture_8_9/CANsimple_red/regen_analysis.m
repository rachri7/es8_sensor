% regen_analysis.m
% Runs the TrueTime model and estimates mean delay using regenerative cycles.

clear global
clear functions
clc

global regen

% Simulation time
simTime = 200;

% Run your Simulink model
% Change CANsimple to the actual model name if different.
sim('CANsimple', simTime);

% Extract regenerative cycle means
cycleMeans = regen.mean;

% Remove empty or invalid values
cycleMeans = cycleMeans(~isnan(cycleMeans));
cycleMeans = cycleMeans(isfinite(cycleMeans));

% Optional: discard first cycles as initial transient
warmupCycles = 5;

if length(cycleMeans) > warmupCycles
    cycleMeansUsed = cycleMeans(warmupCycles+1:end);
else
    cycleMeansUsed = cycleMeans;
end

n = length(cycleMeansUsed);

if n < 1
    error('Not enough regenerative cycles. Increase simTime.');
end

% Estimate mean delay
meanDelay = mean(cycleMeansUsed);

% Standard deviation of cycle means
stdCycles = std(cycleMeansUsed);

% Standard deviation of the mean estimate
stdEstimate = stdCycles / sqrt(n);

% Approximate 95 percent confidence interval using +/- 2 std
ciLow = meanDelay - 2 * stdEstimate;
ciHigh = meanDelay + 2 * stdEstimate;

fprintf('\nRegenerative simulation results:\n');
fprintf('Number of cycles used: %d\n', n);
fprintf('Cycle lengths (packets per cycle):\n');
fprintf('  Min: %d | Max: %d | Mean: %.1f\n', ...
    min(regen.count), max(regen.count), mean(regen.count));
fprintf('Estimated mean delay: %.6f s\n', meanDelay);
fprintf('Std. deviation of cycle means: %.6f s\n', stdCycles);
fprintf('Std. deviation of delay estimate: %.6f s\n', stdEstimate);
fprintf('Approx. 95%% confidence interval: [%.6f, %.6f] s\n', ciLow, ciHigh);

% Optional comparison using all individual delays
if isfield(regen, 'allDelays') && ~isempty(regen.allDelays)
    fprintf('\nIndividual packet delay statistics:\n');
    fprintf('Number of original packets received: %d\n', length(regen.allDelays));
    fprintf('Mean of all packet delays: %.6f s\n', mean(regen.allDelays));
    fprintf('Std. deviation of packet delays: %.6f s\n', std(regen.allDelays));
end