import http from "k6/http";
import { check } from "k6";
import exec from "k6/execution";

function scenarioTemplate(vu, startTime) {
  return {
    executor: "per-vu-iterations",
    vus: vu,
    iterations: 1,
    startTime: startTime,
    gracefulStop: "1m",
  };
}

export const options = {
  scenarios: {
    "1-user": scenarioTemplate(1, "0s"),
    "2-users": scenarioTemplate(2, "30s"),
    "3-users": scenarioTemplate(3, "60s"),
    "4-users": scenarioTemplate(4, "90s"),
    "5-users": scenarioTemplate(5, "120s"),
    "6-users": scenarioTemplate(6, "150s"),
  },
};

const parquets = {
  0: open("./dataset/train/text-to-text-100.json"),
  1: open("./dataset/train/text-to-text-200.json"),
  2: open("./dataset/train/text-to-text-400.json"),
  20: open("dataset/train/20_row.parquet", "b"),
  50: open("dataset/train/50_row.parquet", "b"),
  100: open("dataset/train/100_row.parquet", "b"),
};

const array_of_data = [
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.0001,
      num_train_epochs: 2,
      per_device_eval_batch_size: 2,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[0], "text.json"),
  },
  {
    type: "image",
    params: JSON.stringify({
      resolution: 64,
      train_batch_size: 2,
      num_train_epochs: 2,
      max_train_steps: 50,
      learning_rate: 0.0001,
      gradient_accumulation_steps: 1,
    }),
    file: http.file(parquets[20], "image.parquet"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.0001,
      num_train_epochs: 4,
      per_device_eval_batch_size: 2,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[0], "text.json"),
  },
  {
    type: "image",
    params: JSON.stringify({
      resolution: 64,
      train_batch_size: 2,
      num_train_epochs: 4,
      max_train_steps: 50,
      learning_rate: 0.0001,
      gradient_accumulation_steps: 1,
    }),
    file: http.file(parquets[50], "image.parquet"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.001,
      num_train_epochs: 4,
      per_device_eval_batch_size: 2,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[1], "text.json"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.0001,
      num_train_epochs: 4,
      per_device_eval_batch_size: 2,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[0], "text.json"),
  },
  {
    type: "image",
    params: JSON.stringify({
      resolution: 64,
      train_batch_size: 2,
      num_train_epochs: 4,
      max_train_steps: 50,
      learning_rate: 0.0001,
      gradient_accumulation_steps: 1,
    }),
    file: http.file(parquets[50], "image.parquet"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.001,
      num_train_epochs: 4,
      per_device_eval_batch_size: 2,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[1], "text.json"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.0001,
      num_train_epochs: 4,
      per_device_eval_batch_size: 2,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[0], "text.json"),
  },
  {
    type: "image",
    params: JSON.stringify({
      resolution: 64,
      train_batch_size: 4,
      num_train_epochs: 4,
      max_train_steps: 50,
      learning_rate: 0.0001,
      gradient_accumulation_steps: 2,
    }),
    file: http.file(parquets[100], "image.parquet"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.01,
      num_train_epochs: 4,
      per_device_eval_batch_size: 2,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[1], "text.json"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.001,
      num_train_epochs: 4,
      per_device_eval_batch_size: 2,
      per_device_train_batch_size: 4,
    }),
    file: http.file(parquets[2], "text.json"),
  },
  {
    type: "image",
    params: JSON.stringify({
      resolution: 128,
      train_batch_size: 2,
      num_train_epochs: 4,
      max_train_steps: 50,
      learning_rate: 0.001,
      gradient_accumulation_steps: 2,
    }),
    file: http.file(parquets[20], "image.parquet"),
  },
  {
    type: "image",
    params: JSON.stringify({
      resolution: 256,
      train_batch_size: 4,
      num_train_epochs: 4,
      max_train_steps: 50,
      learning_rate: 0.0001,
      gradient_accumulation_steps: 2,
    }),
    file: http.file(parquets[20], "image.parquet"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.01,
      num_train_epochs: 2,
      per_device_eval_batch_size: 4,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[0], "text.json"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.001,
      num_train_epochs: 6,
      per_device_eval_batch_size: 4,
      per_device_train_batch_size: 4,
    }),
    file: http.file(parquets[1], "text.json"),
  },
  {
    type: "image",
    params: JSON.stringify({
      resolution: 128,
      train_batch_size: 2,
      num_train_epochs: 4,
      max_train_steps: 100,
      learning_rate: 0.001,
      gradient_accumulation_steps: 2,
    }),
    file: http.file(parquets[50], "image.parquet"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.1,
      num_train_epochs: 4,
      per_device_eval_batch_size: 6,
      per_device_train_batch_size: 4,
    }),
    file: http.file(parquets[2], "text.json"),
  },
  {
    type: "image",
    params: JSON.stringify({
      resolution: 256,
      train_batch_size: 2,
      num_train_epochs: 4,
      max_train_steps: 50,
      learning_rate: 0.001,
      gradient_accumulation_steps: 4,
    }),
    file: http.file(parquets[100], "image.parquet"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.01,
      num_train_epochs: 6,
      per_device_eval_batch_size: 4,
      per_device_train_batch_size: 2,
    }),
    file: http.file(parquets[0], "text.json"),
  },
  {
    type: "text",
    params: JSON.stringify({
      learning_rate: 0.01,
      num_train_epochs: 6,
      per_device_eval_batch_size: 4,
      per_device_train_batch_size: 6,
    }),
    file: http.file(parquets[1], "text.json"),
  },
];

export default function () {
  const { scenario } = exec;
  let req = array_of_data[scenario.iterationInTest];
  const res = http.post("http://35.208.32.246:8000/training", req);

  check(res, { "status was 200": (r) => r.status == 200 });
}
