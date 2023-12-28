#!/bin/bash

while [true]
do
    POD_TO_DELETE=$(kubectl get pods -l app="inference"  --no-headers | awk '{print $1}' | head -n 1)

    # Delete the specific pod
    kubectl delete pod "$POD_TO_DELETE"
    sleep 60
done