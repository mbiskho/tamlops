#!/bin/bash
CURRENTDATE=`date +"%d-%m-%Y"`

k6 run load-testing.js
chaos run experiment.json --journal-path "report/chaos-journal-(${CURRENTDATE}).json" --hypothesis-strategy before-method-only
chaos report --export-format=pdf "report/chaos-journal-(${CURRENTDATE}).json" "report/chaos-report-(${CURRENTDATE}).pdf"