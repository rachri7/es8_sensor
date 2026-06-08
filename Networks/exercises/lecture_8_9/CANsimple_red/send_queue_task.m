function [exectime, data] = send_queue_task(seg, data)

global msgQueue
global origInFlight

switch seg

    case 1
        % Simulated execution time of sender task
        exectime = 0.02;

        % If queue is not empty, send one packet
        if ~isempty(msgQueue)

            % Pop first packet from queue
            msg = msgQueue(1);
            msgQueue(1) = [];
            
            %%%%%%%%%%%%%%%%%%%%%%% New part %%%%%%%%%%%%%%%%%%

            % The packet is now submitted to the network, but not yet received
            origInFlight = origInFlight + 1;
        
            % Do not decide cycle end here.
            % The receiver will close the cycle when this packet has arrived
            % and no original packets remain queued or in flight.
            msg.lastInCycle = false;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Send original packet to receiver node 2 on network 1
            ttSendMsg([1 2], msg, data.packetLength, data.priority);
        end

        % Schedule next periodic send
        ttCreateJob('send_queue_task', ttCurrentTime + data.sendPeriod);

    case 2
        exectime = -1;

end