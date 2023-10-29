#!/bin/bash

# Echo
echo "-----------------------------------------------------------------------------"
echo "Experiment has been start!"
echo -e "Semoga eksperimen yang dilakukan akan berjalan dengan lancar aamiin\n"

# Init variables
CURRENTDATE=`date +"%d-%m-%Y-%H:%M"`
CHAOS_DATE="(${CURRENTDATE})"

# Init variables for directories
CHAOS_RUNNING_LOG_DIR="log/running"
CHAOS_FINAL_LOG_DIR="log/${CURRENTDATE}"
CHAOS_REPORT_LOG_DIR="report/${CURRENTDATE}"

# Init variables for directory files
CHAOS_EXPERIMENT_LOG_FILES_DIR="${CHAOS_RUNNING_LOG_DIR}/chaos-experiment.log"
CHAOS_PROXY_REPORT_FILES_DIR="${CHAOS_REPORT_LOG_DIR}/chaos-proxy-journal.json"
CHAOS_COMBINED_REPORT_FILES_DIR="${CHAOS_REPORT_LOG_DIR}/chaos-report.pdf"

# Init commands before testing
rm -r "${CHAOS_RUNNING_LOG_DIR}"
mkdir "${CHAOS_RUNNING_LOG_DIR}"

# Run load testing
export CHAOS_TYPE="steadystate"
k6 run load-testing.js

# Init commands before chaos experiment
mkdir "${CHAOS_REPORT_LOG_DIR}"

# Run chaos experiment: turn off interface-proxy
# gcloud compute instances stop interface-proxy
# sleep 5
export CHAOS_TYPE="off-proxy"
chaos --log-file "${CHAOS_EXPERIMENT_LOG_FILES_DIR}" run experiment/off-proxy.json --journal-path "${CHAOS_PROXY_REPORT_FILES_DIR}" --hypothesis-strategy before-method-only
# gcloud compute instances start interface-proxy
# sleep 5

# Run chaos experiment: turn off event-receiver

# Run chaos experiment: turn off MaaS


# Run chaos experiment: turn off ML Pipeline

# Create reports from chaos experiments
chaos --log-file "${CHAOS_EXPERIMENT_LOG_FILES_DIR}" report --export-format=pdf "${CHAOS_PROXY_REPORT_FILES_DIR}" "${CHAOS_COMBINED_REPORT_FILES_DIR}"

# Move all log from log/running to log/${date}
rm -r "${CHAOS_FINAL_LOG_DIR}"
mkdir "${CHAOS_FINAL_LOG_DIR}"
mv "${CHAOS_RUNNING_LOG_DIR}"/* "${CHAOS_FINAL_LOG_DIR}"
rm -r "${CHAOS_RUNNING_LOG_DIR}"

echo -e "\nExperiment has been done!"
echo "-----------------------------------------------------------------------------"