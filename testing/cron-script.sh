#!/bin/bash
k6 run load-testing.js
chaos run experiment.json --hypothesis-strategy before-method-only