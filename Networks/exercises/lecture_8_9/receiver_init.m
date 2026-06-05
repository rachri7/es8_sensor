function receiver_init

% Initialize TrueTime kernel
ttInitKernel('prioFP');  % scheduling policy - fixed priority

deadline = 10.0;
ttCreateTask('receiver_task', deadline, 'receiver_code');

% Network handler 
ttAttachNetworkHandler(1,'receiver_task')

