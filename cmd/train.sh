#!/bin/bash

# Set the deployment name
DEPLOYMENT_NAME="bismakhomeini-trainning"

while true
do
    kubectl get pods -l app="$DEPLOYMENT_NAME" -o wide >> pod.csv
    kubectl get pods -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,CONTAINERS_READY:.status.containerStatuses[*].ready,POD_SCHEDULED:.status.conditions[?(@.reason=='PodScheduled')].status,READY:.status.conditions[?(@.type=='Ready')].status,READINESS:.status.conditions[?(@.type=='Ready')].lastTransitionTime,LIVENESS:.status.conditions[?(@.type=='Ready')].lastProbeTime,RESTART_COUNT:.status.containerStatuses[*].restartCount,LAST_STATE:.status.containerStatuses[*].lastState,STARTED:.status.startTime,FINISHED:.status.containerStatuses[*].state.terminated.finishedAt" >> info.csv
    nvidia-smi --query-gpu=timestamp,name,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> gpu.csv
    echo "Added for GPU and Etc"
done