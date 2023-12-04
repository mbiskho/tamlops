import http from 'k6/http';
import { check } from 'k6';
import exec from 'k6/execution';

export const options = {
  iterations: 1,
  vus: 10
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
  const { vu } = exec;
  const learningRate = [0.001, 0.01, 0.0001]
  const resolution = [32, 64, 128]
  const evalBatch = 2 * Math.round(Math.random() * 2 + 2)
  let req = {}

  if (vu.iterationInInstance % 10 < 5) {
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
    console.log(req.params)
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
      file: vu.iterationInInstance < 7 ? http.file(parquets[20], '20_row.parquet') : vu.iterationInInstance < 9 ? http.file(parquets[50], '50_row.parquet') : http.file(parquets[100], '100_row.parquet')
    }
  }

  const res = http.post('http://35.208.32.246:8000/training', req)

  check(res, { 'status was 200': (r) => r.status == 200 });
}