function [exectime, data] = highprio_code(seg, data)

switch seg

    case 1
        % Simulated execution time
        exectime = 0.00001;

        % Create high-priority message
        msg.ttime = ttCurrentTime;
        msg.payload = rand();
        msg.source = 3;              % high-priority node ID
        msg.cycle = 0;               % not used for regenerative estimate
        msg.lastInCycle = false;
        msg.seq = 0;

        % Send high-priority packet to receiver node 2
        ttSendMsg(data.receiver, msg, data.packetLength, data.priority);

        % Generate next Poisson arrival time
        u = rand();
        T = -log(1-u) / data.lambda;        %ICDF (Inverse function method)

        % Schedule next high-priority packet
        ttCreateJob('highprio_task', ttCurrentTime + T);

    case 2
        exectime = -1;

end