import http from 'k6/http';
import { check, sleep } from 'k6';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.4/index.js';

export const options = {
  vus: 10,
  stages: [
    // { duration: '30s', target: 20 },
    // { duration: '1m30s', target: 10 },
    { duration: '30s', target: 10 },
    { duration: '30s', target: 0 },
  ],
};

export default function () {
  // const res = http.get('http://138.197.231.156:8000/health');

  let data = {
    "type": "text",
    "text": "What german name of apple"
  };
  const res = http.post('http://138.197.231.156:8000/inference', JSON.stringify(data))

  check(res, { 'status was 200': (r) => r.status == 200 });
  sleep(15);
}

export function handleSummary(data) {
  return {
    [`log/running/load-testing-${__ENV.CHAOS_TYPE}.log`]: textSummary(data, { indent: '→', enableColors: false })
  };
}