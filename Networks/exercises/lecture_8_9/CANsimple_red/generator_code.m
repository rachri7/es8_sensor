function [exectime, data] = generator_code(seg, data)

global msgQueue
global currentCycle originalSeq regen

switch seg

    case 1
        % Simulated execution time of packet generation
        exectime = 0.005;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% New part %%%%%%%%%%%%%%%%%%%%%%%%
        % Increase packet sequence number
        originalSeq = originalSeq + 1;

        % Initialize sum and count arrays for a new regenerative cycle
        if length(regen.sum) < currentCycle
            regen.sum(currentCycle) = 0;
            regen.count(currentCycle) = 0;
        end

        % Create packet
        msg.ttime = ttCurrentTime;     % generation timestamp
        msg.payload = rand();          % optional payload
        msg.source = 1;                % original node ID
        msg.cycle = currentCycle;      % regenerative cycle number
        msg.lastInCycle = false;       % filled in by sender task if needed
        msg.seq = originalSeq;         % packet sequence number
        
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Add packet to queue
        if isempty(msgQueue)
            msgQueue = msg;
        else
            msgQueue(end+1) = msg;
        end

        % Random inter-arrival time, exponential distribution
        u = rand();
        T = -log(1-u) / data.lambda;    %ICDF (Inverse function method)

        % Schedule next packet generation
        ttCreateJob('generator_task', ttCurrentTime + T);

    case 2
        exectime = -1;

end