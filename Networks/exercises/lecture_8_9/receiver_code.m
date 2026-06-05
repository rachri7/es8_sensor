function [exectime, data] = receiver_code(seg,data)


    msg = ttGetMsg(1);
    if(length(msg)>0)
       ttAnalogOut(1,ttCurrentTime-msg(1));
    else
       %ttAnalogOut(1,ttCurrentTime);
    end
    exectime = -1;


    
