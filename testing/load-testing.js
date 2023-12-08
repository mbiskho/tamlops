import http from 'k6/http';
import { check, sleep } from 'k6';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.4/index.js';
import exec from 'k6/execution';
import { SharedArray } from 'k6/data';

export const options = {
  vus: 0,
  stages: [
    { duration: '5m', target: 128 },
    { duration: '10m', target: 128 },
    { duration: '5m', target: 0 }
  ],
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

  const res = http.post('http://35.208.32.246:8000/inference', JSON.stringify(data))

  check(res, { 'status was 200': (r) => r.status == 200 });
  sleep(15);
}

// export function handleSummary(data) {
//   return {
//     [`log/running/load-testing-${__ENV.CHAOS_TYPE}.log`]: textSummary(data, { indent: '→', enableColors: false })
//   };
// }
