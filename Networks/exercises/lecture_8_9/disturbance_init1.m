function disturbance_init1

data=1;
% Initialize TrueTime kernel
ttInitKernel('prioFP');  % scheduling policy - fixed priority
ttCreateTask('generator_task',1,'generator_code',data);
ttCreateJob('generator_task',ttCurrentTime);


