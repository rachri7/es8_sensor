function [exectime, data] = receiver_code(seg, data)

global regen
global msgQueue
global currentCycle
global origInFlight

switch seg
    case 1
        msg = ttGetMsg(1);

        if ~isempty(msg)

            if msg(1).source == 1

                % Original packet has now arrived at the receiver,
                % so it is no longer in flight on the CAN network.
                origInFlight = origInFlight - 1;

                % Delay from original packet generation to reception
                delay = ttCurrentTime - msg(1).ttime;

                % Output original-node delay to scope
                ttAnalogOut(1, delay);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Store individual delay
                regen.allDelays(end+1) = delay;

                % Cycle number assigned when packet was generated
                c = msg(1).cycle;

                % Initialize storage if needed
                if length(regen.sum) < c
                    regen.sum(c) = 0;
                    regen.count(c) = 0;
                end

                % Accumulate delay statistics for this cycle
                regen.sum(c) = regen.sum(c) + delay;
                regen.count(c) = regen.count(c) + 1;

                % Close cycle only when:
                % 1) transmitter queue is empty
                % 2) no original packets are still in the CAN network
                if isempty(msgQueue) && origInFlight == 0

                    % Store cycle mean delay
                    if regen.count(c) > 0
                        regen.mean(end+1) = regen.sum(c) / regen.count(c);
                    end

                    % Start new regenerative cycle
                    currentCycle = currentCycle + 1;
                else
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end
            end
        end

        exectime = -1;
end
