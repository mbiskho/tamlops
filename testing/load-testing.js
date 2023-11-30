import http from 'k6/http';
import { check, sleep } from 'k6';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js'
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.4/index.js';
import exec from 'k6/execution';

export const options = {
  vus: 0,
  stages: [
    { duration: '30s', target: 600 },
    { duration: '120s', target: 600 },
    { duration: '30s', target: 0 }
  ],
};

const DATASETS = new SharedArray('prompts', function () {
  return JSON.parse(open('./dataset/combined-dataset-55k.json'));
});

export default function () {
  const { vu } = exec;
  let data = {}

  if (vu.iterationInInstance % 600 < 540)
    data = DATASETS[Math.floor(Math.random() * 50000)]
  else data = DATASETS[Math.floor(Math.random() * 5000 + 50000)]

  const res = http.post('http://35.208.32.246:8000/inference', JSON.stringify(data))

  check(res, { 'status was 200': (r) => r.status == 200 });
  sleep(15);
}

export function handleSummary(data) {
  return {
    [`log/running/load-testing-${__ENV.CHAOS_TYPE}.log`]: textSummary(data, { indent: '→', enableColors: false })
  };
}
