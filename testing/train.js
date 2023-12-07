import http from 'k6/http';
import { check } from 'k6';
import exec from 'k6/execution';

function scenarioTemplate(vu, startTime) {
  return {
    executor: 'per-vu-iterations',
    vus: vu,
    iterations: 1,
    startTime: startTime,
    gracefulStop: '1m'
  }
}

export const options = {
  scenarios: {
    '1-user': {
      ...scenarioTemplate(1, '0s')
    },
    '2-users': {
      ...scenarioTemplate(2, '30s')
    },
    '3-users': {
      ...scenarioTemplate(3, '60s')
    },
    '4-users': {
      ...scenarioTemplate(4, '90s')
    },
    '5-users': {
      ...scenarioTemplate(5, '120s')
    },
    '6-users': {
      ...scenarioTemplate(6, '150s')
    },
  },
};

const parquets = {
  0: open('./dataset/train/text-to-text-100.json'),
  1: open('./dataset/train/text-to-text-200.json'),
  2: open('./dataset/train/text-to-text-400.json'),
  20: open('dataset/train/20_row.parquet', 'b'),
  50: open('dataset/train/50_row.parquet', 'b'),
  100: open('dataset/train/100_row.parquet', 'b')
}

export default function () {
  const { scenario } = exec;
  const learningRate = [0.001, 0.01, 0.0001]
  const resolution = [32, 64, 128]
  const evalBatch = 2 * Math.round(Math.random() * 2 + 2)
  let req = {}

  if (scenario.iterationInTest % 2 === 0) {
    req = {
      type: 'text',
      params: JSON.stringify({
        "learning_rate": learningRate[Math.round(Math.random() * 2)],
        "num_train_epochs": Math.round(Math.random() * 4 + 2),
        "per_device_eval_batch_size": evalBatch,
        "per_device_train_batch_size": evalBatch
      }),
      file: http.file(parquets[Math.round(Math.random() * 2)], 'text.json')
    }
  }
  else {
    req = {
      type: 'image',
      params: JSON.stringify({
        "resolution": resolution[Math.round(Math.random() * 2)],
        "train_batch_size": evalBatch,
        "num_train_epochs": Math.round(Math.random() * 4 + 2),
        "max_train_steps": 50 * Math.round(Math.random() * 2 + 1),
        "learning_rate": learningRate[Math.round(Math.random() * 2)],
        "gradient_accumulation_steps": 1 * Math.round(Math.random() * 2 + 1)
      }),
      file: scenario.iterationInTest < 7 ? http.file(parquets[20], '20_row.parquet') : scenario.iterationInTest < 9 ? http.file(parquets[50], '50_row.parquet') : http.file(parquets[100], '100_row.parquet')
    }
  }

  const res = http.post('http://35.208.32.246:8000/training', req)

  check(res, { 'status was 200': (r) => r.status == 200 });
}