function highprio_init

% Initialize TrueTime kernel
ttInitKernel('prioFP');

% High-priority node parameters
data.dataRate = 12500;       % CAN data rate [bits/s], must match network block in simulink
data.packetLength = 250;      % high-priority packet length [bits]
data.priority = 0;            % highest CAN priority
data.receiver = [1 2];        % network 1, receiver node 2

% Original node traffic parameters.
% These should match transmitter_init.m.
originalPacketLength = 250;
originalSendPeriod = 0.035;

% Target total bus utilization
targetUtilization = 0.95;

% Original node bandwidth usage [bits/s]
originalBitRate = originalPacketLength / originalSendPeriod;

% Remaining bandwidth for high-priority node [bits/s]
% For total network utilization 95%
%highPriorityBitRate = targetUtilization * data.dataRate - originalBitRate;
% High priority node 95% utilization
highPriorityBitRate = targetUtilization * data.dataRate;

% Convert bandwidth to Poisson packet rate
data.lambda = highPriorityBitRate / data.packetLength;

% Safety check
if data.lambda <= 0
    error('High-priority lambda is <= 0. Check dataRate, packetLength, and sendPeriod.');
end

disp(['High-priority lambda = ', num2str(data.lambda), ' packets/s']);

% Create high-priority Poisson sender task
ttCreateTask('highprio_task', 1, 'highprio_code', data);

% Start immediately
ttCreateJob('highprio_task', ttCurrentTime);