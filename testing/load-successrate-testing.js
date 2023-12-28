import http from 'k6/http';
import { check, sleep } from 'k6';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.4/index.js';
import exec from 'k6/execution';
import { SharedArray } from 'k6/data';

function scenarioTemplate(vu, startTime) {
  return {
    executor: 'per-vu-iterations',
    vus: vu,
    iterations: 1,
    maxDuration: '1m',
    startTime: startTime,
    gracefulStop: '0s'
  }
}

export const options = {
  scenarios: {
    '1-user': scenarioTemplate(1, '0m'),
    '2-users': scenarioTemplate(2, '1m'),
    '4-users': scenarioTemplate(4, '2m'),
    '8-users': scenarioTemplate(8, '3m'),
    '16-users': scenarioTemplate(16, '4m'),
    '32-users': scenarioTemplate(32, '5m'),
    '64-users': scenarioTemplate(64, '6m'),
    '128-users': scenarioTemplate(128, '7m'),
    '256-users': scenarioTemplate(256, '8m'),
    '512-users': scenarioTemplate(512, '9m')
  },
};

const TEXT_DATASETS = new SharedArray('text-prompts', function () {
  return JSON.parse(open('./dataset/text-to-text-300k.json'));
});

const IMAGE_DATASETS = new SharedArray('image-prompts', function () {
  return JSON.parse(open('./dataset/text-to-image-100k.json'));
});

export default function () {
  const { scenario } = exec;
  let data = {}

  if (scenario.iterationInTest % 19 < 17) {
    let size = TEXT_DATASETS.length
    data = {
      type: 'text',
      text: TEXT_DATASETS[Math.floor(Math.random() * size)].text
    }
  }
  else {
    let size = IMAGE_DATASETS.length
    data = {
      type: 'image',
      text: IMAGE_DATASETS[Math.floor(Math.random() * size)].text
    }
  }

  const res = http.post('http://34.42.105.222:8000/inference/test', JSON.stringify(data))

  check(res, { 'status was 200': (r) => r.status == 200 });
}

export function handleSummary(data) {
  return {
    [`log/success-rate/load-successrate-testing.log`]: textSummary(data, { indent: '→', enableColors: false })
  };
}
