function receiver_init

% Initialize TrueTime kernel
ttInitKernel('prioFP');

deadline = 10.0;

% Create receiver task
ttCreateTask('receiver_task', deadline, 'receiver_code');

% Run receiver task whenever a packet arrives on network 1
ttAttachNetworkHandler(1, 'receiver_task');