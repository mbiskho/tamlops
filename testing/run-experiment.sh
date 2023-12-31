#!/bin/bash

# Run env
source ~/tamlops/testing/env/bin/activate
cd ~/tamlops/testing

# Echo
echo "-----------------------------------------------------------------------------"
echo "Experiment has been start!"
echo -e "Semoga eksperimen yang dilakukan akan berjalan dengan lancar aamiin\n"

# Init variables
CURRENTDATE=`date +"%d-%m-%Y-%H:%M"`
CHAOS_DATE="(${CURRENTDATE})"

# Init variables for directories
CHAOS_RUNNING_LOG_DIR="log/running"
CHAOS_EXPERIMENT_LOG_DIR="experiment"
CHAOS_FINAL_LOG_DIR="log/${CURRENTDATE}"
CHAOS_REPORT_LOG_DIR="report/${CURRENTDATE}"

# Init variables for directory files
CHAOS_INTERFACE_TEST_FILE_DIR="${CHAOS_EXPERIMENT_LOG_DIR}/off-interface.json"
CHAOS_INFERENCE_TEST_FILE_DIR="${CHAOS_EXPERIMENT_LOG_DIR}/off-inference.json"
CHAOS_TRAINING_TEST_FILE_DIR="${CHAOS_EXPERIMENT_LOG_DIR}/off-training.json"

CHAOS_EXPERIMENT_LOG_FILE_DIR="${CHAOS_RUNNING_LOG_DIR}/chaos-experiment.log"

CHAOS_INTERFACE_REPORT_FILE_DIR="${CHAOS_REPORT_LOG_DIR}/chaos-interface-journal.json"
CHAOS_INFERENCE_REPORT_FILE_DIR="${CHAOS_REPORT_LOG_DIR}/chaos-inference-journal.json"
CHAOS_TRAINING_REPORT_FILE_DIR="${CHAOS_REPORT_LOG_DIR}/chaos-training-journal.json"

CHAOS_COMBINED_REPORT_FILES_DIR="${CHAOS_REPORT_LOG_DIR}/chaos-report.md"

# Init commands before testing
rm -r "${CHAOS_RUNNING_LOG_DIR}"
mkdir "${CHAOS_RUNNING_LOG_DIR}"

# Run load testing
export CHAOS_TYPE="steadystate"
K6_PROMETHEUS_RW_TREND_AS_NATIVE_HISTOGRAM=true k6 run -o experimental-prometheus-rw load-testing.js

# Init commands before chaos experiment
mkdir "${CHAOS_REPORT_LOG_DIR}"

# Run chaos experiment: turn off interface
export CHAOS_TYPE="off-interface"
chaos --log-file "${CHAOS_EXPERIMENT_LOG_FILE_DIR}" run "${CHAOS_INTERFACE_TEST_FILE_DIR}" --journal-path "${CHAOS_INTERFACE_REPORT_FILE_DIR}" --hypothesis-strategy before-method-only
# Run chaos experiment: turn off ML Inference
export CHAOS_TYPE="off-inference"
chaos --log-file "${CHAOS_EXPERIMENT_LOG_FILE_DIR}" run "${CHAOS_INFERENCE_TEST_FILE_DIR}" --journal-path "${CHAOS_INFERENCE_REPORT_FILE_DIR}" --hypothesis-strategy before-method-only
# Run chaos experiment: turn off ML Training
export CHAOS_TYPE="off-training"
chaos --log-file "${CHAOS_EXPERIMENT_LOG_FILE_DIR}" run "${CHAOS_TRAINING_TEST_FILE_DIR}" --journal-path "${CHAOS_TRAINING_REPORT_FILE_DIR}" --hypothesis-strategy before-method-only

# Create reports from chaos experiments
chaos --log-file "${CHAOS_EXPERIMENT_LOG_FILE_DIR}" report --export-format=md "${CHAOS_INTERFACE_REPORT_FILE_DIR}" "${CHAOS_INFERENCE_REPORT_FILE_DIR}" "${CHAOS_TRAINING_REPORT_FILE_DIR}" "${CHAOS_COMBINED_REPORT_FILES_DIR}"

# Move all log from log/running to log/${date}
rm -r "${CHAOS_FINAL_LOG_DIR}"
mkdir "${CHAOS_FINAL_LOG_DIR}"
mv "${CHAOS_RUNNING_LOG_DIR}"/* "${CHAOS_FINAL_LOG_DIR}"
rm -r "${CHAOS_RUNNING_LOG_DIR}"

echo -e "\nExperiment has been done!"
echo "-----------------------------------------------------------------------------"
