#!/bin/bash

# run prometheus + node-exporter
systemctl daemon-reload
systemctl enable node_exporter
systemctl start node_exporter
systemctl enable prometheus
systemctl start prometheus

# run service
uvicorn  app:app --host 0.0.0.0 --port 8000 --workers 2 --reload