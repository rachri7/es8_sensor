function [exectime, data] = disturbance_code(seg,dataL)

networkNbr = 1;  % select a random network (1-3)
msg = [];                        % empty message
priority = 0;                    % highest priority
lambda=30;
u=rand();
msg = [ttCurrentTime]; 
T=1;
data=1;

switch seg
  case 1
    exectime = 0.005;
    ttSendMsg([1 2], msg, 250, priority);
    ttCreateJob('generator_task',ttCurrentTime+T);
  case 2
     exectime = -1;
end


