#!/bin/bash

# run prometheus + node-exporter
systemctl daemon-reload
systemctl enable node_exporter
systemctl start node_exporter
systemctl enable prometheus
systemctl start prometheus

# run service
python3 init.py