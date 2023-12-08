---
date: 08 December 2023
title: Chaos Engineering Report
---

-   [Summary](#summary){#toc-summary}
-   [Experiments](#experiments){#toc-experiments}
    -   [Chaos Offline ML Inference Server
        Experiment](#chaos-offline-ml-inference-server-experiment){#toc-chaos-offline-ml-inference-server-experiment}
        -   [Summary](#summary-1){#toc-summary-1}
        -   [Definition](#definition){#toc-definition}
        -   [Result](#result){#toc-result}
        -   [Appendix](#appendix){#toc-appendix}
    -   [Chaos Offline ML Training Server
        Experiment](#chaos-offline-ml-training-server-experiment){#toc-chaos-offline-ml-training-server-experiment}
        -   [Summary](#summary-2){#toc-summary-2}
        -   [Definition](#definition-1){#toc-definition-1}
        -   [Result](#result-1){#toc-result-1}
        -   [Appendix](#appendix-1){#toc-appendix-1}
    -   [Chaos Offline ML Training Server
        Experiment](#chaos-offline-ml-training-server-experiment-1){#toc-chaos-offline-ml-training-server-experiment-1}
        -   [Summary](#summary-3){#toc-summary-3}
        -   [Definition](#definition-2){#toc-definition-2}
        -   [Result](#result-2){#toc-result-2}
        -   [Appendix](#appendix-2){#toc-appendix-2}

```{=tex}
\newpage
```
# Summary

This report aggregates 3 experiments spanning over the following
subjects:

*Inference*, *RTX 2080*, *Kubernetes*, *Docker*, *DGX-A100*,
*Interface*, *Training*, *Pod*, *Compute Engine*, *Google Cloud
Platform*

```{=tex}
\newpage
```
# Experiments

## Chaos Offline ML Inference Server Experiment

This experiment is to test the load testing performance & find the
errors when ML Inference is offline (offline)

### Summary

Chaos Offline ML Inference Server Experiment

This experiment is to test the load testing performance & find the
errors when ML Inference is offline (offline)

  ------------------------------------------ ---------------------------------------------
  **Status**                                 completed

  **Tagged**                                 Kubernetes, Pod, RTX 2080, Inference

  **Executed From**                          srv420659

  **Platform**                               Linux-5.15.0-1047-kvm-x86_64-with-glibc2.35

  **Started**                                Fri, 08 Dec 2023 18:08:09 GMT

  **Completed**                              Fri, 08 Dec 2023 18:14:50 GMT

  **Duration**                               6 minutes
  ------------------------------------------ ---------------------------------------------

### Definition

The experiment was made of 5 actions, to vary conditions in your system,
and 0 probes, to collect objective data from your system during the
experiment.

#### Steady State Hypothesis

The steady state hypothesis this experiment tried was "**Make sure that
load testing has been done & able to prompt every types to server**".

##### Before Run

The steady state was verified

  ------------------------------------------------------------------------------
  Probe                                          Tolerance            Verified
  ---------------------------------------------- -------------------- ----------
  Normal load success rate testing log must       True                True
  exists                                                              

  Normal load testing log must exists             True                True

  We can request text                             200                 True

  We can request image                            200                 True
  ------------------------------------------------------------------------------

##### After Run

The steady state was not verified. 

  ------------------------------------------------------------------------------
  Probe                                          Tolerance            Verified
  ---------------------------------------------- -------------------- ----------

  ------------------------------------------------------------------------------

#### Method

The experiment method defines the sequence of activities that help
gathering evidence towards, or against, the hypothesis.

The following activities were conducted as part of the experimental's
method:

  -----------------------------------------------------------------------
  Type      Name
  --------- -------------------------------------------------------------
  action     Sleep to give time for turning off the Inference pod

  action     Run load success rate testing

  action     Run load testing

  action     Run load testing

  action     Sleep to give time for turning on the Inference pods
  -----------------------------------------------------------------------

### Result

The experiment was conducted on Fri, 08 Dec 2023 18:08:09 GMT and lasted
roughly 6 minutes.

#### Action - Sleep to give time for turning off the Inference pod

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:08:10 GMT
  **Ended**        Fri, 08 Dec 2023 18:09:10 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        sleep

  **Timeout**     N/A

  **Arguments**   60
  --------------- --------------------------------------------------------

#### Action - Run load success rate testing

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:09:10 GMT
  **Ended**        Fri, 08 Dec 2023 18:10:04 GMT
  **Duration**     54 seconds
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        k6

  **Timeout**     N/A

  **Arguments**   run -o experimental-prometheus-rw
                  load-successrate-testing.js
  --------------- --------------------------------------------------------

#### Action - Run load testing

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:10:04 GMT
  **Ended**        Fri, 08 Dec 2023 18:11:54 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        k6

  **Timeout**     N/A

  **Arguments**   run -o experimental-prometheus-rw load-testing.js
  --------------- --------------------------------------------------------

#### Action - Run load testing

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:11:54 GMT
  **Ended**        Fri, 08 Dec 2023 18:13:50 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        k6

  **Timeout**     N/A

  **Arguments**   run -o experimental-prometheus-rw load-testing.js
  --------------- --------------------------------------------------------

#### Action - Sleep to give time for turning on the Inference pods

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:13:50 GMT
  **Ended**        Fri, 08 Dec 2023 18:14:50 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        sleep

  **Timeout**     N/A

  **Arguments**   60
  --------------- --------------------------------------------------------

### Appendix

#### Action - Sleep to give time for turning off the Inference pod

The *action* returned the following result:

``` javascript
{'status': 0, 'stderr': '', 'stdout': ''}
```

#### Action - Run load success rate testing

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': '',
 'stdout': '\n'
           '          /\\      |‾‾| /‾‾/   /‾‾/   \n'
           '     /\\  /  \\     |  |/  /   /  /    \n'
           '    /  \\/    \\    |     (   /   ‾‾\\  \n'
           '   /          \\   |  |\\  \\ |  (‾)  | \n'
           '  / __________ \\  |__| \\__\\ \\_____/ .io\n'
           '\n'
           '  execution: local\n'
           '     script: load-successrate-testing.js\n'
           '     output: Prometheus remote write '
           '(http://localhost:9090/api/v1/write)\n'
           '\n'
           '  scenarios: (100.00%) 1 scenario, 1 max VUs, 1m0s max duration '
           '(incl. graceful stop):\n'
           '           * 1-user: 1 iterations for each of 1 VUs (maxDuration: '
           '1m0s)\n'
           '\n'
           '\n'
           'running (0m01.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m01.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m02.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m02.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m03.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m03.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m04.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m04.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m05.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m05.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m06.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m06.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m07.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m07.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m08.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m08.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m09.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m09.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m10.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m10.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m11.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m11.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m12.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m12.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m13.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m13.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m14.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m14.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m15.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m15.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m15.7s), 0/1 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           '1-user ✓ [ 100% ] 1 VUs  0m15.7s/1m0s  1/1 iters, 1 per VU\n'}
```

#### Action - Run load testing

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': '',
 'stdout': '\n'
           '          /\\      |‾‾| /‾‾/   /‾‾/   \n'
           '     /\\  /  \\     |  |/  /   /  /    \n'
           '    /  \\/    \\    |     (   /   ‾‾\\  \n'
           '   /          \\   |  |\\  \\ |  (‾)  | \n'
           '  / __________ \\  |__| \\__\\ \\_____/ .io\n'
           '\n'
           '  execution: local\n'
           '     script: load-testing.js\n'
           '     output: Prometheus remote write '
           '(http://localhost:9090/api/v1/write)\n'
           '\n'
           '  scenarios: (100.00%) 1 scenario, 16 max VUs, 1m30s max duration '
           '(incl. graceful stop):\n'
           '           * default: Up to 16 looping VUs for 1m0s over 1 stages '
           '(gracefulRampDown: 30s, gracefulStop: 30s)\n'
           '\n'
           '\n'
           'running (0m01.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   2% ] 00/16 VUs  0m01.0s/1m00.0s\n'
           '\n'
           'running (0m02.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   3% ] 00/16 VUs  0m02.0s/1m00.0s\n'
           '\n'
           'running (0m03.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   5% ] 00/16 VUs  0m03.0s/1m00.0s\n'
           '\n'
           'running (0m04.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   7% ] 01/16 VUs  0m04.0s/1m00.0s\n'
           '\n'
           'running (0m05.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   8% ] 01/16 VUs  0m05.0s/1m00.0s\n'
           '\n'
           'running (0m06.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  10% ] 01/16 VUs  0m06.0s/1m00.0s\n'
           '\n'
           'running (0m07.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  12% ] 01/16 VUs  0m07.0s/1m00.0s\n'
           '\n'
           'running (0m08.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  13% ] 02/16 VUs  0m08.0s/1m00.0s\n'
           '\n'
           'running (0m09.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  15% ] 02/16 VUs  0m09.0s/1m00.0s\n'
           '\n'
           'running (0m10.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  17% ] 02/16 VUs  0m10.0s/1m00.0s\n'
           '\n'
           'running (0m11.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  18% ] 02/16 VUs  0m11.0s/1m00.0s\n'
           '\n'
           'running (0m12.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  20% ] 03/16 VUs  0m12.0s/1m00.0s\n'
           '\n'
           'running (0m13.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  22% ] 03/16 VUs  0m13.0s/1m00.0s\n'
           '\n'
           'running (0m14.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  23% ] 03/16 VUs  0m14.0s/1m00.0s\n'
           '\n'
           'running (0m15.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  25% ] 03/16 VUs  0m15.0s/1m00.0s\n'
           '\n'
           'running (0m16.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  27% ] 04/16 VUs  0m16.0s/1m00.0s\n'
           '\n'
           'running (0m17.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  28% ] 04/16 VUs  0m17.0s/1m00.0s\n'
           '\n'
           'running (0m18.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  30% ] 04/16 VUs  0m18.0s/1m00.0s\n'
           '\n'
           'running (0m19.0s), 05/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  32% ] 05/16 VUs  0m19.0s/1m00.0s\n'
           '\n'
           'running (0m20.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  33% ] 05/16 VUs  0m20.0s/1m00.0s\n'
           '\n'
           'running (0m21.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  35% ] 05/16 VUs  0m21.0s/1m00.0s\n'
           '\n'
           'running (0m22.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  37% ] 05/16 VUs  0m22.0s/1m00.0s\n'
           '\n'
           'running (0m23.0s), 06/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  38% ] 06/16 VUs  0m23.0s/1m00.0s\n'
           '\n'
           'running (0m24.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  40% ] 06/16 VUs  0m24.0s/1m00.0s\n'
           '\n'
           'running (0m25.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  42% ] 06/16 VUs  0m25.0s/1m00.0s\n'
           '\n'
           'running (0m26.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  43% ] 06/16 VUs  0m26.0s/1m00.0s\n'
           '\n'
           'running (0m27.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  45% ] 07/16 VUs  0m27.0s/1m00.0s\n'
           '\n'
           'running (0m28.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  47% ] 07/16 VUs  0m28.0s/1m00.0s\n'
           '\n'
           'running (0m29.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  48% ] 07/16 VUs  0m29.0s/1m00.0s\n'
           '\n'
           'running (0m30.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  50% ] 07/16 VUs  0m30.0s/1m00.0s\n'
           '\n'
           'running (0m31.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  52% ] 08/16 VUs  0m31.0s/1m00.0s\n'
           '\n'
           'running (0m32.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  53% ] 08/16 VUs  0m32.0s/1m00.0s\n'
           '\n'
           'running (0m33.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  55% ] 08/16 VUs  0m33.0s/1m00.0s\n'
           '\n'
           'running (0m34.0s), 09/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  57% ] 09/16 VUs  0m34.0s/1m00.0s\n'
           '\n'
           'running (0m35.0s), 09/16 VUs, 5 complete and 0 interrupted '
           'iterations\n'
           'default   [  58% ] 09/16 VUs  0m35.0s/1m00.0s\n'
           '\n'
           'running (0m36.0s), 09/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  60% ] 09/16 VUs  0m36.0s/1m00.0s\n'
           '\n'
           'running (0m37.0s), 09/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  62% ] 09/16 VUs  0m37.0s/1m00.0s\n'
           '\n'
           'running (0m38.0s), 10/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  63% ] 10/16 VUs  0m38.0s/1m00.0s\n'
           '\n'
           'running (0m39.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  65% ] 10/16 VUs  0m39.0s/1m00.0s\n'
           '\n'
           'running (0m40.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  67% ] 10/16 VUs  0m40.0s/1m00.0s\n'
           '\n'
           'running (0m41.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  68% ] 10/16 VUs  0m41.0s/1m00.0s\n'
           '\n'
           'running (0m42.0s), 11/16 VUs, 9 complete and 0 interrupted '
           'iterations\n'
           'default   [  70% ] 11/16 VUs  0m42.0s/1m00.0s\n'
           '\n'
           'running (0m43.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  72% ] 11/16 VUs  0m43.0s/1m00.0s\n'
           '\n'
           'running (0m44.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  73% ] 11/16 VUs  0m44.0s/1m00.0s\n'
           '\n'
           'running (0m45.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  75% ] 11/16 VUs  0m45.0s/1m00.0s\n'
           '\n'
           'running (0m46.0s), 12/16 VUs, 11 complete and 0 interrupted '
           'iterations\n'
           'default   [  77% ] 12/16 VUs  0m46.0s/1m00.0s\n'
           '\n'
           'running (0m47.0s), 12/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  78% ] 12/16 VUs  0m47.0s/1m00.0s\n'
           '\n'
           'running (0m48.0s), 12/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  80% ] 12/16 VUs  0m48.0s/1m00.0s\n'
           '\n'
           'running (0m49.0s), 13/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  82% ] 13/16 VUs  0m49.0s/1m00.0s\n'
           '\n'
           'running (0m50.0s), 13/16 VUs, 13 complete and 0 interrupted '
           'iterations\n'
           'default   [  83% ] 13/16 VUs  0m50.0s/1m00.0s\n'
           '\n'
           'running (0m51.0s), 13/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  85% ] 13/16 VUs  0m51.0s/1m00.0s\n'
           '\n'
           'running (0m52.0s), 13/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  87% ] 13/16 VUs  0m52.0s/1m00.0s\n'
           '\n'
           'running (0m53.0s), 14/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  88% ] 14/16 VUs  0m53.0s/1m00.0s\n'
           '\n'
           'running (0m54.0s), 14/16 VUs, 17 complete and 0 interrupted '
           'iterations\n'
           'default   [  90% ] 14/16 VUs  0m54.0s/1m00.0s\n'
           '\n'
           'running (0m55.0s), 14/16 VUs, 18 complete and 0 interrupted '
           'iterations\n'
           'default   [  92% ] 14/16 VUs  0m55.0s/1m00.0s\n'
           '\n'
           'running (0m56.0s), 14/16 VUs, 18 complete and 0 interrupted '
           'iterations\n'
           'default   [  93% ] 14/16 VUs  0m56.0s/1m00.0s\n'
           '\n'
           'running (0m57.0s), 15/16 VUs, 19 complete and 0 interrupted '
           'iterations\n'
           'default   [  95% ] 15/16 VUs  0m57.0s/1m00.0s\n'
           '\n'
           'running (0m58.0s), 15/16 VUs, 20 complete and 0 interrupted '
           'iterations\n'
           'default   [  97% ] 15/16 VUs  0m58.0s/1m00.0s\n'
           '\n'
           'running (0m59.0s), 15/16 VUs, 21 complete and 0 interrupted '
           'iterations\n'
           'default   [  98% ] 15/16 VUs  0m59.0s/1m00.0s\n'
           '\n'
           'running (1m00.0s), 15/16 VUs, 21 complete and 0 interrupted '
           'iterations\n'
           'default   [ 100% ] 15/16 VUs  1m00.0s/1m00.0s\n'
           '\n'
           'running (1m01.0s), 14/16 VUs, 22 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m02.0s), 13/16 VUs, 23 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m03.0s), 12/16 VUs, 24 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m04.0s), 12/16 VUs, 24 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m05.0s), 11/16 VUs, 25 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m06.0s), 09/16 VUs, 27 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m07.0s), 08/16 VUs, 28 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m08.0s), 08/16 VUs, 28 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m09.0s), 06/16 VUs, 30 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m10.0s), 05/16 VUs, 31 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m11.0s), 04/16 VUs, 32 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m12.0s), 03/16 VUs, 33 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m13.0s), 02/16 VUs, 34 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 16/16 VUs  1m0s\n'
           '\n'
           'running (1m13.9s), 00/16 VUs, 36 complete and 0 interrupted '
           'iterations\n'
           'default ✓ [ 100% ] 00/16 VUs  1m0s\n'}
```

#### Action - Run load testing

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': '',
 'stdout': '\n'
           '          /\\      |‾‾| /‾‾/   /‾‾/   \n'
           '     /\\  /  \\     |  |/  /   /  /    \n'
           '    /  \\/    \\    |     (   /   ‾‾\\  \n'
           '   /          \\   |  |\\  \\ |  (‾)  | \n'
           '  / __________ \\  |__| \\__\\ \\_____/ .io\n'
           '\n'
           '  execution: local\n'
           '     script: load-testing.js\n'
           '     output: Prometheus remote write '
           '(http://localhost:9090/api/v1/write)\n'
           '\n'
           '  scenarios: (100.00%) 1 scenario, 16 max VUs, 1m30s max duration '
           '(incl. graceful stop):\n'
           '           * default: Up to 16 looping VUs for 1m0s over 1 stages '
           '(gracefulRampDown: 30s, gracefulStop: 30s)\n'
           '\n'
           '\n'
           'running (0m01.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   2% ] 00/16 VUs  0m01.0s/1m00.0s\n'
           '\n'
           'running (0m02.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   3% ] 00/16 VUs  0m02.0s/1m00.0s\n'
           '\n'
           'running (0m03.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   5% ] 00/16 VUs  0m03.0s/1m00.0s\n'
           '\n'
           'running (0m04.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   7% ] 01/16 VUs  0m04.0s/1m00.0s\n'
           '\n'
           'running (0m05.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   8% ] 01/16 VUs  0m05.0s/1m00.0s\n'
           '\n'
           'running (0m06.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  10% ] 01/16 VUs  0m06.0s/1m00.0s\n'
           '\n'
           'running (0m07.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  12% ] 01/16 VUs  0m07.0s/1m00.0s\n'
           '\n'
           'running (0m08.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  13% ] 02/16 VUs  0m08.0s/1m00.0s\n'
           '\n'
           'running (0m09.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  15% ] 02/16 VUs  0m09.0s/1m00.0s\n'
           '\n'
           'running (0m10.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  17% ] 02/16 VUs  0m10.0s/1m00.0s\n'
           '\n'
           'running (0m11.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  18% ] 02/16 VUs  0m11.0s/1m00.0s\n'
           '\n'
           'running (0m12.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  20% ] 03/16 VUs  0m12.0s/1m00.0s\n'
           '\n'
           'running (0m13.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  22% ] 03/16 VUs  0m13.0s/1m00.0s\n'
           '\n'
           'running (0m14.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  23% ] 03/16 VUs  0m14.0s/1m00.0s\n'
           '\n'
           'running (0m15.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  25% ] 03/16 VUs  0m15.0s/1m00.0s\n'
           '\n'
           'running (0m16.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  27% ] 04/16 VUs  0m16.0s/1m00.0s\n'
           '\n'
           'running (0m17.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  28% ] 04/16 VUs  0m17.0s/1m00.0s\n'
           '\n'
           'running (0m18.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  30% ] 04/16 VUs  0m18.0s/1m00.0s\n'
           '\n'
           'running (0m19.0s), 05/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  32% ] 05/16 VUs  0m19.0s/1m00.0s\n'
           '\n'
           'running (0m20.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  33% ] 05/16 VUs  0m20.0s/1m00.0s\n'
           '\n'
           'running (0m21.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  35% ] 05/16 VUs  0m21.0s/1m00.0s\n'
           '\n'
           'running (0m22.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  37% ] 05/16 VUs  0m22.0s/1m00.0s\n'
           '\n'
           'running (0m23.0s), 06/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  38% ] 06/16 VUs  0m23.0s/1m00.0s\n'
           '\n'
           'running (0m24.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  40% ] 06/16 VUs  0m24.0s/1m00.0s\n'
           '\n'
           'running (0m25.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  42% ] 06/16 VUs  0m25.0s/1m00.0s\n'
           '\n'
           'running (0m26.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  43% ] 06/16 VUs  0m26.0s/1m00.0s\n'
           '\n'
           'running (0m27.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  45% ] 07/16 VUs  0m27.0s/1m00.0s\n'
           '\n'
           'running (0m28.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  47% ] 07/16 VUs  0m28.0s/1m00.0s\n'
           '\n'
           'running (0m29.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  48% ] 07/16 VUs  0m29.0s/1m00.0s\n'
           '\n'
           'running (0m30.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  50% ] 07/16 VUs  0m30.0s/1m00.0s\n'
           '\n'
           'running (0m31.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  52% ] 08/16 VUs  0m31.0s/1m00.0s\n'
           '\n'
           'running (0m32.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  53% ] 08/16 VUs  0m32.0s/1m00.0s\n'
           '\n'
           'running (0m33.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  55% ] 08/16 VUs  0m33.0s/1m00.0s\n'
           '\n'
           'running (0m34.0s), 09/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  57% ] 09/16 VUs  0m34.0s/1m00.0s\n'
           '\n'
           'running (0m35.0s), 09/16 VUs, 5 complete and 0 interrupted '
           'iterations\n'
           'default   [  58% ] 09/16 VUs  0m35.0s/1m00.0s\n'
           '\n'
           'running (0m36.0s), 09/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  60% ] 09/16 VUs  0m36.0s/1m00.0s\n'
           '\n'
           'running (0m37.0s), 09/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  62% ] 09/16 VUs  0m37.0s/1m00.0s\n'
           '\n'
           'running (0m38.0s), 10/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  63% ] 10/16 VUs  0m38.0s/1m00.0s\n'
           '\n'
           'running (0m39.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  65% ] 10/16 VUs  0m39.0s/1m00.0s\n'
           '\n'
           'running (0m40.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  67% ] 10/16 VUs  0m40.0s/1m00.0s\n'
           '\n'
           'running (0m41.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  68% ] 10/16 VUs  0m41.0s/1m00.0s\n'
           '\n'
           'running (0m42.0s), 11/16 VUs, 9 complete and 0 interrupted '
           'iterations\n'
           'default   [  70% ] 11/16 VUs  0m42.0s/1m00.0s\n'
           '\n'
           'running (0m43.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  72% ] 11/16 VUs  0m43.0s/1m00.0s\n'
           '\n'
           'running (0m44.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  73% ] 11/16 VUs  0m44.0s/1m00.0s\n'
           '\n'
           'running (0m45.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  75% ] 11/16 VUs  0m45.0s/1m00.0s\n'
           '\n'
           'running (0m46.0s), 12/16 VUs, 11 complete and 0 interrupted '
           'iterations\n'
           'default   [  77% ] 12/16 VUs  0m46.0s/1m00.0s\n'
           '\n'
           'running (0m47.0s), 12/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  78% ] 12/16 VUs  0m47.0s/1m00.0s\n'
           '\n'
           'running (0m48.0s), 12/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  80% ] 12/16 VUs  0m48.0s/1m00.0s\n'
           '\n'
           'running (0m49.0s), 13/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  82% ] 13/16 VUs  0m49.0s/1m00.0s\n'
           '\n'
           'running (0m50.0s), 13/16 VUs, 13 complete and 0 interrupted '
           'iterations\n'
           'default   [  83% ] 13/16 VUs  0m50.0s/1m00.0s\n'
           '\n'
           'running (0m51.0s), 13/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  85% ] 13/16 VUs  0m51.0s/1m00.0s\n'
           '\n'
           'running (0m52.0s), 13/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  87% ] 13/16 VUs  0m52.0s/1m00.0s\n'
           '\n'
           'running (0m53.0s), 14/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  88% ] 14/16 VUs  0m53.0s/1m00.0s\n'
           '\n'
           'running (0m54.0s), 14/16 VUs, 17 complete and 0 interrupted '
           'iterations\n'
           'default   [  90% ] 14/16 VUs  0m54.0s/1m00.0s\n'
           '\n'
           'running (0m55.0s), 14/16 VUs, 18 complete and 0 interrupted '
           'iterations\n'
           'default   [  92% ] 14/16 VUs  0m55.0s/1m00.0s\n'
           '\n'
           'running (0m56.0s), 14/16 VUs, 18 complete and 0 interrupted '
           'iterations\n'
           'default   [  93% ] 14/16 VUs  0m56.0s/1m00.0s\n'
           '\n'
           'running (0m57.0s), 15/16 VUs, 19 complete and 0 interrupted '
           'iterations\n'
           'default   [  95% ] 15/16 VUs  0m57.0s/1m00.0s\n'
           '\n'
           'running (0m58.0s), 15/16 VUs, 20 complete and 0 interrupted '
           'iterations\n'
           'default   [  97% ] 15/16 VUs  0m58.0s/1m00.0s\n'
           '\n'
           'running (0m59.0s), 15/16 VUs, 21 complete and 0 interrupted '
           'iterations\n'
           'default   [  98% ] 15/16 VUs  0m59.0s/1m00.0s\n'
           '\n'
           'running (1m00.0s), 15/16 VUs, 21 complete and 0 interrupted '
           'iterations\n'
           'default   [ 100% ] 15/16 VUs  1m00.0s/1m00.0s\n'
           '\n'
           'running (1m01.0s), 14/16 VUs, 22 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m02.0s), 13/16 VUs, 23 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m03.0s), 12/16 VUs, 24 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m04.0s), 12/16 VUs, 24 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m05.0s), 11/16 VUs, 25 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m06.0s), 09/16 VUs, 27 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m07.0s), 08/16 VUs, 28 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m08.0s), 08/16 VUs, 28 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m09.0s), 06/16 VUs, 30 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m10.0s), 05/16 VUs, 31 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m11.0s), 04/16 VUs, 32 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m12.0s), 03/16 VUs, 33 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m13.0s), 02/16 VUs, 34 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m13.9s), 00/16 VUs, 36 complete and 0 interrupted '
           'iterations\n'
           'default ✓ [ 100% ] 00/16 VUs  1m0s\n'}
```

#### Action - Sleep to give time for turning on the Inference pods

The *action* returned the following result:

``` javascript
{'status': 0, 'stderr': '', 'stdout': ''}
```

## Chaos Offline ML Training Server Experiment

This experiment is to test the load testing performance & find the
errors when ML Inference is offline (offline)

### Summary

Chaos Offline ML Training Server Experiment

This experiment is to test the load testing performance & find the
errors when ML Inference is offline (offline)

  ------------------------------------------ ---------------------------------------------
  **Status**                                 completed

  **Tagged**                                 Google Cloud Platform, Compute Engine,
                                             Docker, Interface

  **Executed From**                          srv420659

  **Platform**                               Linux-5.15.0-1047-kvm-x86_64-with-glibc2.35

  **Started**                                Fri, 08 Dec 2023 18:02:08 GMT

  **Completed**                              Fri, 08 Dec 2023 18:08:03 GMT

  **Duration**                               5 minutes
  ------------------------------------------ ---------------------------------------------

### Definition

The experiment was made of 5 actions, to vary conditions in your system,
and 0 probes, to collect objective data from your system during the
experiment.

#### Steady State Hypothesis

The steady state hypothesis this experiment tried was "**Make sure that
load testing has been done & able to prompt every types to server**".

##### Before Run

The steady state was verified

  ------------------------------------------------------------------------------
  Probe                                          Tolerance            Verified
  ---------------------------------------------- -------------------- ----------
  Normal load success rate testing log must       True                True
  exists                                                              

  Normal load testing log must exists             True                True

  We can request text                             200                 True

  We can request image                            200                 True
  ------------------------------------------------------------------------------

##### After Run

The steady state was not verified. 

  ------------------------------------------------------------------------------
  Probe                                          Tolerance            Verified
  ---------------------------------------------- -------------------- ----------

  ------------------------------------------------------------------------------

#### Method

The experiment method defines the sequence of activities that help
gathering evidence towards, or against, the hypothesis.

The following activities were conducted as part of the experimental's
method:

  -----------------------------------------------------------------------
  Type      Name
  --------- -------------------------------------------------------------
  action     Turn off interface VM on Google Cloud Platform

  action     Run load success rate testing

  action     Run load testing

  action     Turn on interface VM on Google Cloud Platform

  action     Turn on docker instance
  -----------------------------------------------------------------------

### Result

The experiment was conducted on Fri, 08 Dec 2023 18:02:08 GMT and lasted
roughly 5 minutes.

#### Action - Turn off interface VM on Google Cloud Platform

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:02:09 GMT
  **Ended**        Fri, 08 Dec 2023 18:03:11 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        gcloud

  **Timeout**     N/A

  **Arguments**   compute instances stop --zone us-central1-a
                  proxy-interaface
  --------------- --------------------------------------------------------

#### Action - Run load success rate testing

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:03:11 GMT
  **Ended**        Fri, 08 Dec 2023 18:04:36 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        k6

  **Timeout**     N/A

  **Arguments**   run -o experimental-prometheus-rw
                  load-successrate-testing.js
  --------------- --------------------------------------------------------

#### Action - Run load testing

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:04:36 GMT
  **Ended**        Fri, 08 Dec 2023 18:06:46 GMT
  **Duration**     2 minutes
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        k6

  **Timeout**     N/A

  **Arguments**   run -o experimental-prometheus-rw load-testing.js
  --------------- --------------------------------------------------------

#### Action - Turn on interface VM on Google Cloud Platform

  ------------------ -------------------------------
  **Status**         succeeded
  **Background**     False
  **Started**        Fri, 08 Dec 2023 18:06:46 GMT
  **Ended**          Fri, 08 Dec 2023 18:06:59 GMT
  **Duration**       13 seconds
  **Paused After**   60s
  ------------------ -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        gcloud

  **Timeout**     N/A

  **Arguments**   compute instances start --zone us-central1-a
                  proxy-interaface
  --------------- --------------------------------------------------------

#### Action - Turn on docker instance

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:07:59 GMT
  **Ended**        Fri, 08 Dec 2023 18:08:03 GMT
  **Duration**     4 seconds
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        ssh

  **Timeout**     N/A

  **Arguments**   -i gcp-ta-key muhammad_haqqi01@35.208.32.246 sudo docker
                  container start interface_proxy_1
  --------------- --------------------------------------------------------

### Appendix

#### Action - Turn off interface VM on Google Cloud Platform

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': 'Stopping instance(s) proxy-interaface...\n'
           '......................................................................................................................................................................................................................................................................................done.\n'
           'Updated '
           '[https://compute.googleapis.com/compute/v1/projects/mlops-398205/zones/us-central1-a/instances/proxy-interaface].\n',
 'stdout': ''}
```

#### Action - Run load success rate testing

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': 'time="2023-12-08T18:04:19Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n',
 'stdout': '\n'
           '          /\\      |‾‾| /‾‾/   /‾‾/   \n'
           '     /\\  /  \\     |  |/  /   /  /    \n'
           '    /  \\/    \\    |     (   /   ‾‾\\  \n'
           '   /          \\   |  |\\  \\ |  (‾)  | \n'
           '  / __________ \\  |__| \\__\\ \\_____/ .io\n'
           '\n'
           '  execution: local\n'
           '     script: load-successrate-testing.js\n'
           '     output: Prometheus remote write '
           '(http://localhost:9090/api/v1/write)\n'
           '\n'
           '  scenarios: (100.00%) 1 scenario, 1 max VUs, 1m0s max duration '
           '(incl. graceful stop):\n'
           '           * 1-user: 1 iterations for each of 1 VUs (maxDuration: '
           '1m0s)\n'
           '\n'
           '\n'
           'running (0m01.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m01.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m02.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m02.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m03.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m03.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m04.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m04.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m05.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m05.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m06.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m06.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m07.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m07.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m08.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m08.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m09.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m09.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m10.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m10.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m11.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m11.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m12.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m12.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m13.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m13.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m14.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m14.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m15.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m15.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m16.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m16.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m17.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m17.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m18.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m18.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m19.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m19.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m20.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m20.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m21.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m21.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m22.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m22.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m23.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m23.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m24.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m24.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m25.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m25.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m26.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m26.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m27.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m27.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m28.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m28.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m29.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m29.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m30.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m30.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m31.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m31.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m32.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m32.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m33.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m33.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m34.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m34.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m35.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m35.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m36.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m36.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m37.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m37.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m38.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m38.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m39.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m39.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m40.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m40.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m41.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m41.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m42.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m42.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m43.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m43.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m44.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m44.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m45.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m45.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m45.0s), 0/1 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           '1-user ✓ [ 100% ] 1 VUs  0m45.0s/1m0s  1/1 iters, 1 per VU\n'}
```

#### Action - Run load testing

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': 'time="2023-12-08T18:05:48Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:05:52Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:05:56Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:05:59Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:03Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:07Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:11Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:14Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:18Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:22Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:26Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:29Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:33Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:33Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:37Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:37Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:41Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n'
           'time="2023-12-08T18:06:41Z" level=warning msg="Request Failed" '
           'error="Post \\"http://35.208.32.246:8000/inference\\": dial: i/o '
           'timeout"\n',
 'stdout': '\n'
           '          /\\      |‾‾| /‾‾/   /‾‾/   \n'
           '     /\\  /  \\     |  |/  /   /  /    \n'
           '    /  \\/    \\    |     (   /   ‾‾\\  \n'
           '   /          \\   |  |\\  \\ |  (‾)  | \n'
           '  / __________ \\  |__| \\__\\ \\_____/ .io\n'
           '\n'
           '  execution: local\n'
           '     script: load-testing.js\n'
           '     output: Prometheus remote write '
           '(http://localhost:9090/api/v1/write)\n'
           '\n'
           '  scenarios: (100.00%) 1 scenario, 16 max VUs, 1m30s max duration '
           '(incl. graceful stop):\n'
           '           * default: Up to 16 looping VUs for 1m0s over 1 stages '
           '(gracefulRampDown: 30s, gracefulStop: 30s)\n'
           '\n'
           '\n'
           'running (0m01.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   2% ] 00/16 VUs  0m01.0s/1m00.0s\n'
           '\n'
           'running (0m02.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   3% ] 00/16 VUs  0m02.0s/1m00.0s\n'
           '\n'
           'running (0m03.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   5% ] 00/16 VUs  0m03.0s/1m00.0s\n'
           '\n'
           'running (0m04.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   7% ] 01/16 VUs  0m04.0s/1m00.0s\n'
           '\n'
           'running (0m05.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   8% ] 01/16 VUs  0m05.0s/1m00.0s\n'
           '\n'
           'running (0m06.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  10% ] 01/16 VUs  0m06.0s/1m00.0s\n'
           '\n'
           'running (0m07.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  12% ] 01/16 VUs  0m07.0s/1m00.0s\n'
           '\n'
           'running (0m08.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  13% ] 02/16 VUs  0m08.0s/1m00.0s\n'
           '\n'
           'running (0m09.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  15% ] 02/16 VUs  0m09.0s/1m00.0s\n'
           '\n'
           'running (0m10.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  17% ] 02/16 VUs  0m10.0s/1m00.0s\n'
           '\n'
           'running (0m11.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  18% ] 02/16 VUs  0m11.0s/1m00.0s\n'
           '\n'
           'running (0m12.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  20% ] 03/16 VUs  0m12.0s/1m00.0s\n'
           '\n'
           'running (0m13.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  22% ] 03/16 VUs  0m13.0s/1m00.0s\n'
           '\n'
           'running (0m14.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  23% ] 03/16 VUs  0m14.0s/1m00.0s\n'
           '\n'
           'running (0m15.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  25% ] 03/16 VUs  0m15.0s/1m00.0s\n'
           '\n'
           'running (0m16.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  27% ] 04/16 VUs  0m16.0s/1m00.0s\n'
           '\n'
           'running (0m17.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  28% ] 04/16 VUs  0m17.0s/1m00.0s\n'
           '\n'
           'running (0m18.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  30% ] 04/16 VUs  0m18.0s/1m00.0s\n'
           '\n'
           'running (0m19.0s), 05/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  32% ] 05/16 VUs  0m19.0s/1m00.0s\n'
           '\n'
           'running (0m20.0s), 05/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  33% ] 05/16 VUs  0m20.0s/1m00.0s\n'
           '\n'
           'running (0m21.0s), 05/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  35% ] 05/16 VUs  0m21.0s/1m00.0s\n'
           '\n'
           'running (0m22.0s), 05/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  37% ] 05/16 VUs  0m22.0s/1m00.0s\n'
           '\n'
           'running (0m23.0s), 06/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  38% ] 06/16 VUs  0m23.0s/1m00.0s\n'
           '\n'
           'running (0m24.0s), 06/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  40% ] 06/16 VUs  0m24.0s/1m00.0s\n'
           '\n'
           'running (0m25.0s), 06/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  42% ] 06/16 VUs  0m25.0s/1m00.0s\n'
           '\n'
           'running (0m26.0s), 06/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  43% ] 06/16 VUs  0m26.0s/1m00.0s\n'
           '\n'
           'running (0m27.0s), 07/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  45% ] 07/16 VUs  0m27.0s/1m00.0s\n'
           '\n'
           'running (0m28.0s), 07/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  47% ] 07/16 VUs  0m28.0s/1m00.0s\n'
           '\n'
           'running (0m29.0s), 07/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  48% ] 07/16 VUs  0m29.0s/1m00.0s\n'
           '\n'
           'running (0m30.0s), 07/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  50% ] 07/16 VUs  0m30.0s/1m00.0s\n'
           '\n'
           'running (0m31.0s), 08/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  52% ] 08/16 VUs  0m31.0s/1m00.0s\n'
           '\n'
           'running (0m32.0s), 08/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  53% ] 08/16 VUs  0m32.0s/1m00.0s\n'
           '\n'
           'running (0m33.0s), 08/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  55% ] 08/16 VUs  0m33.0s/1m00.0s\n'
           '\n'
           'running (0m34.0s), 09/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  57% ] 09/16 VUs  0m34.0s/1m00.0s\n'
           '\n'
           'running (0m35.0s), 09/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  58% ] 09/16 VUs  0m35.0s/1m00.0s\n'
           '\n'
           'running (0m36.0s), 09/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  60% ] 09/16 VUs  0m36.0s/1m00.0s\n'
           '\n'
           'running (0m37.0s), 09/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  62% ] 09/16 VUs  0m37.0s/1m00.0s\n'
           '\n'
           'running (0m38.0s), 10/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  63% ] 10/16 VUs  0m38.0s/1m00.0s\n'
           '\n'
           'running (0m39.0s), 10/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  65% ] 10/16 VUs  0m39.0s/1m00.0s\n'
           '\n'
           'running (0m40.0s), 10/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  67% ] 10/16 VUs  0m40.0s/1m00.0s\n'
           '\n'
           'running (0m41.0s), 10/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  68% ] 10/16 VUs  0m41.0s/1m00.0s\n'
           '\n'
           'running (0m42.0s), 11/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  70% ] 11/16 VUs  0m42.0s/1m00.0s\n'
           '\n'
           'running (0m43.0s), 11/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  72% ] 11/16 VUs  0m43.0s/1m00.0s\n'
           '\n'
           'running (0m44.0s), 11/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  73% ] 11/16 VUs  0m44.0s/1m00.0s\n'
           '\n'
           'running (0m45.0s), 11/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  75% ] 11/16 VUs  0m45.0s/1m00.0s\n'
           '\n'
           'running (0m46.0s), 12/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  77% ] 12/16 VUs  0m46.0s/1m00.0s\n'
           '\n'
           'running (0m47.0s), 12/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  78% ] 12/16 VUs  0m47.0s/1m00.0s\n'
           '\n'
           'running (0m48.0s), 12/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  80% ] 12/16 VUs  0m48.0s/1m00.0s\n'
           '\n'
           'running (0m49.0s), 13/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  82% ] 13/16 VUs  0m49.0s/1m00.0s\n'
           '\n'
           'running (0m50.0s), 13/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  83% ] 13/16 VUs  0m50.0s/1m00.0s\n'
           '\n'
           'running (0m51.0s), 13/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  85% ] 13/16 VUs  0m51.0s/1m00.0s\n'
           '\n'
           'running (0m52.0s), 13/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  87% ] 13/16 VUs  0m52.0s/1m00.0s\n'
           '\n'
           'running (0m53.0s), 14/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  88% ] 14/16 VUs  0m53.0s/1m00.0s\n'
           '\n'
           'running (0m54.0s), 14/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  90% ] 14/16 VUs  0m54.0s/1m00.0s\n'
           '\n'
           'running (0m55.0s), 14/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  92% ] 14/16 VUs  0m55.0s/1m00.0s\n'
           '\n'
           'running (0m56.0s), 14/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  93% ] 14/16 VUs  0m56.0s/1m00.0s\n'
           '\n'
           'running (0m57.0s), 15/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  95% ] 15/16 VUs  0m57.0s/1m00.0s\n'
           '\n'
           'running (0m58.0s), 15/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  97% ] 15/16 VUs  0m58.0s/1m00.0s\n'
           '\n'
           'running (0m59.0s), 15/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  98% ] 15/16 VUs  0m59.0s/1m00.0s\n'
           '\n'
           'running (1m00.0s), 15/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [ 100% ] 15/16 VUs  1m00.0s/1m00.0s\n'
           '\n'
           'running (1m01.0s), 14/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m02.0s), 14/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m03.0s), 14/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m04.0s), 13/16 VUs, 5 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m05.0s), 13/16 VUs, 5 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m06.0s), 13/16 VUs, 5 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m07.0s), 13/16 VUs, 5 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m08.0s), 12/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m09.0s), 12/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m10.0s), 12/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m11.0s), 12/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m12.0s), 11/16 VUs, 7 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m13.0s), 11/16 VUs, 7 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m14.0s), 11/16 VUs, 7 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m15.0s), 11/16 VUs, 7 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m16.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m17.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m18.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m19.0s), 09/16 VUs, 9 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m20.0s), 09/16 VUs, 9 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m21.0s), 09/16 VUs, 9 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m22.0s), 09/16 VUs, 9 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m23.0s), 08/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m24.0s), 08/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m25.0s), 08/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m26.0s), 08/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m27.0s), 07/16 VUs, 11 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m28.0s), 07/16 VUs, 11 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m29.0s), 07/16 VUs, 11 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m30.0s), 07/16 VUs, 11 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m30.0s), 00/16 VUs, 11 complete and 7 interrupted '
           'iterations\n'
           'default ✓ [ 100% ] 04/16 VUs  1m0s\n'}
```

#### Action - Turn on interface VM on Google Cloud Platform

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': 'Starting instance(s) proxy-interaface...\n'
           '....................................done.\n'
           'Updated '
           '[https://compute.googleapis.com/compute/v1/projects/mlops-398205/zones/us-central1-a/instances/proxy-interaface].\n'
           'Instance internal IP is 10.128.0.14\n'
           'Instance external IP is 35.208.32.246\n',
 'stdout': ''}
```

#### Action - Turn on docker instance

The *action* returned the following result:

``` javascript
{'status': 0, 'stderr': '', 'stdout': 'interface_proxy_1\n'}
```

## Chaos Offline ML Training Server Experiment

This experiment is to test the load testing performance & find the
errors when ML Training is offline (offline)

### Summary

Chaos Offline ML Training Server Experiment

This experiment is to test the load testing performance & find the
errors when ML Training is offline (offline)

  ------------------------------------------ ---------------------------------------------
  **Status**                                 completed

  **Tagged**                                 Kubernetes, Pod, DGX-A100, Training

  **Executed From**                          srv420659

  **Platform**                               Linux-5.15.0-1047-kvm-x86_64-with-glibc2.35

  **Started**                                Fri, 08 Dec 2023 18:14:54 GMT

  **Completed**                              Fri, 08 Dec 2023 18:19:41 GMT

  **Duration**                               4 minutes
  ------------------------------------------ ---------------------------------------------

### Definition

The experiment was made of 4 actions, to vary conditions in your system,
and 0 probes, to collect objective data from your system during the
experiment.

#### Steady State Hypothesis

The steady state hypothesis this experiment tried was "**Make sure that
load testing has been done & able to prompt every types to server**".

##### Before Run

The steady state was verified

  ------------------------------------------------------------------------------
  Probe                                          Tolerance            Verified
  ---------------------------------------------- -------------------- ----------
  Normal load success rate testing log must       True                True
  exists                                                              

  Normal load testing log must exists             True                True

  We can request text                             200                 True

  We can request image                            200                 True
  ------------------------------------------------------------------------------

##### After Run

The steady state was not verified. 

  ------------------------------------------------------------------------------
  Probe                                          Tolerance            Verified
  ---------------------------------------------- -------------------- ----------

  ------------------------------------------------------------------------------

#### Method

The experiment method defines the sequence of activities that help
gathering evidence towards, or against, the hypothesis.

The following activities were conducted as part of the experimental's
method:

  -----------------------------------------------------------------------
  Type      Name
  --------- -------------------------------------------------------------
  action     Sleep to give time for turning off the Training pod

  action     Run load success rate testing

  action     Run load testing

  action     Sleep to give time for turning on all Training pods
  -----------------------------------------------------------------------

### Result

The experiment was conducted on Fri, 08 Dec 2023 18:14:54 GMT and lasted
roughly 4 minutes.

#### Action - Sleep to give time for turning off the Training pod

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:14:55 GMT
  **Ended**        Fri, 08 Dec 2023 18:15:55 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        sleep

  **Timeout**     N/A

  **Arguments**   60
  --------------- --------------------------------------------------------

#### Action - Run load success rate testing

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:15:55 GMT
  **Ended**        Fri, 08 Dec 2023 18:16:49 GMT
  **Duration**     54 seconds
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        k6

  **Timeout**     N/A

  **Arguments**   run -o experimental-prometheus-rw
                  load-successrate-testing.js
  --------------- --------------------------------------------------------

#### Action - Run load testing

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:16:49 GMT
  **Ended**        Fri, 08 Dec 2023 18:18:41 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        k6

  **Timeout**     N/A

  **Arguments**   run -o experimental-prometheus-rw load-testing.js
  --------------- --------------------------------------------------------

#### Action - Sleep to give time for turning on all Training pods

  ---------------- -------------------------------
  **Status**       succeeded
  **Background**   False
  **Started**      Fri, 08 Dec 2023 18:18:41 GMT
  **Ended**        Fri, 08 Dec 2023 18:19:41 GMT
  **Duration**     1 minute
  ---------------- -------------------------------

The action provider that was executed:

  --------------- --------------------------------------------------------
  **Type**        process

  **Path**        sleep

  **Timeout**     N/A

  **Arguments**   60
  --------------- --------------------------------------------------------

### Appendix

#### Action - Sleep to give time for turning off the Training pod

The *action* returned the following result:

``` javascript
{'status': 0, 'stderr': '', 'stdout': ''}
```

#### Action - Run load success rate testing

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': '',
 'stdout': '\n'
           '          /\\      |‾‾| /‾‾/   /‾‾/   \n'
           '     /\\  /  \\     |  |/  /   /  /    \n'
           '    /  \\/    \\    |     (   /   ‾‾\\  \n'
           '   /          \\   |  |\\  \\ |  (‾)  | \n'
           '  / __________ \\  |__| \\__\\ \\_____/ .io\n'
           '\n'
           '  execution: local\n'
           '     script: load-successrate-testing.js\n'
           '     output: Prometheus remote write '
           '(http://localhost:9090/api/v1/write)\n'
           '\n'
           '  scenarios: (100.00%) 1 scenario, 1 max VUs, 1m0s max duration '
           '(incl. graceful stop):\n'
           '           * 1-user: 1 iterations for each of 1 VUs (maxDuration: '
           '1m0s)\n'
           '\n'
           '\n'
           'running (0m01.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m01.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m02.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m02.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m03.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m03.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m04.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m04.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m05.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m05.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m06.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m06.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m07.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m07.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m08.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m08.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m09.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m09.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m10.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m10.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m11.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m11.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m12.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m12.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m13.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m13.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m14.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m14.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m15.0s), 1/1 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           '1-user   [   0% ] 1 VUs  0m15.0s/1m0s  0/1 iters, 1 per VU\n'
           '\n'
           'running (0m15.7s), 0/1 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           '1-user ✓ [ 100% ] 1 VUs  0m15.7s/1m0s  1/1 iters, 1 per VU\n'}
```

#### Action - Run load testing

The *action* returned the following result:

``` javascript
{'status': 0,
 'stderr': '',
 'stdout': '\n'
           '          /\\      |‾‾| /‾‾/   /‾‾/   \n'
           '     /\\  /  \\     |  |/  /   /  /    \n'
           '    /  \\/    \\    |     (   /   ‾‾\\  \n'
           '   /          \\   |  |\\  \\ |  (‾)  | \n'
           '  / __________ \\  |__| \\__\\ \\_____/ .io\n'
           '\n'
           '  execution: local\n'
           '     script: load-testing.js\n'
           '     output: Prometheus remote write '
           '(http://localhost:9090/api/v1/write)\n'
           '\n'
           '  scenarios: (100.00%) 1 scenario, 16 max VUs, 1m30s max duration '
           '(incl. graceful stop):\n'
           '           * default: Up to 16 looping VUs for 1m0s over 1 stages '
           '(gracefulRampDown: 30s, gracefulStop: 30s)\n'
           '\n'
           '\n'
           'running (0m01.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   2% ] 00/16 VUs  0m01.0s/1m00.0s\n'
           '\n'
           'running (0m02.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   3% ] 00/16 VUs  0m02.0s/1m00.0s\n'
           '\n'
           'running (0m03.0s), 00/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   5% ] 00/16 VUs  0m03.0s/1m00.0s\n'
           '\n'
           'running (0m04.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   7% ] 01/16 VUs  0m04.0s/1m00.0s\n'
           '\n'
           'running (0m05.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [   8% ] 01/16 VUs  0m05.0s/1m00.0s\n'
           '\n'
           'running (0m06.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  10% ] 01/16 VUs  0m06.0s/1m00.0s\n'
           '\n'
           'running (0m07.0s), 01/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  12% ] 01/16 VUs  0m07.0s/1m00.0s\n'
           '\n'
           'running (0m08.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  13% ] 02/16 VUs  0m08.0s/1m00.0s\n'
           '\n'
           'running (0m09.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  15% ] 02/16 VUs  0m09.0s/1m00.0s\n'
           '\n'
           'running (0m10.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  17% ] 02/16 VUs  0m10.0s/1m00.0s\n'
           '\n'
           'running (0m11.0s), 02/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  18% ] 02/16 VUs  0m11.0s/1m00.0s\n'
           '\n'
           'running (0m12.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  20% ] 03/16 VUs  0m12.0s/1m00.0s\n'
           '\n'
           'running (0m13.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  22% ] 03/16 VUs  0m13.0s/1m00.0s\n'
           '\n'
           'running (0m14.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  23% ] 03/16 VUs  0m14.0s/1m00.0s\n'
           '\n'
           'running (0m15.0s), 03/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  25% ] 03/16 VUs  0m15.0s/1m00.0s\n'
           '\n'
           'running (0m16.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  27% ] 04/16 VUs  0m16.0s/1m00.0s\n'
           '\n'
           'running (0m17.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  28% ] 04/16 VUs  0m17.0s/1m00.0s\n'
           '\n'
           'running (0m18.0s), 04/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  30% ] 04/16 VUs  0m18.0s/1m00.0s\n'
           '\n'
           'running (0m19.0s), 05/16 VUs, 0 complete and 0 interrupted '
           'iterations\n'
           'default   [  32% ] 05/16 VUs  0m19.0s/1m00.0s\n'
           '\n'
           'running (0m20.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  33% ] 05/16 VUs  0m20.0s/1m00.0s\n'
           '\n'
           'running (0m21.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  35% ] 05/16 VUs  0m21.0s/1m00.0s\n'
           '\n'
           'running (0m22.0s), 05/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  37% ] 05/16 VUs  0m22.0s/1m00.0s\n'
           '\n'
           'running (0m23.0s), 06/16 VUs, 1 complete and 0 interrupted '
           'iterations\n'
           'default   [  38% ] 06/16 VUs  0m23.0s/1m00.0s\n'
           '\n'
           'running (0m24.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  40% ] 06/16 VUs  0m24.0s/1m00.0s\n'
           '\n'
           'running (0m25.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  42% ] 06/16 VUs  0m25.0s/1m00.0s\n'
           '\n'
           'running (0m26.0s), 06/16 VUs, 2 complete and 0 interrupted '
           'iterations\n'
           'default   [  43% ] 06/16 VUs  0m26.0s/1m00.0s\n'
           '\n'
           'running (0m27.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  45% ] 07/16 VUs  0m27.0s/1m00.0s\n'
           '\n'
           'running (0m28.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  47% ] 07/16 VUs  0m28.0s/1m00.0s\n'
           '\n'
           'running (0m29.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  48% ] 07/16 VUs  0m29.0s/1m00.0s\n'
           '\n'
           'running (0m30.0s), 07/16 VUs, 3 complete and 0 interrupted '
           'iterations\n'
           'default   [  50% ] 07/16 VUs  0m30.0s/1m00.0s\n'
           '\n'
           'running (0m31.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  52% ] 08/16 VUs  0m31.0s/1m00.0s\n'
           '\n'
           'running (0m32.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  53% ] 08/16 VUs  0m32.0s/1m00.0s\n'
           '\n'
           'running (0m33.0s), 08/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  55% ] 08/16 VUs  0m33.0s/1m00.0s\n'
           '\n'
           'running (0m34.0s), 09/16 VUs, 4 complete and 0 interrupted '
           'iterations\n'
           'default   [  57% ] 09/16 VUs  0m34.0s/1m00.0s\n'
           '\n'
           'running (0m35.0s), 09/16 VUs, 5 complete and 0 interrupted '
           'iterations\n'
           'default   [  58% ] 09/16 VUs  0m35.0s/1m00.0s\n'
           '\n'
           'running (0m36.0s), 09/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  60% ] 09/16 VUs  0m36.0s/1m00.0s\n'
           '\n'
           'running (0m37.0s), 09/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  62% ] 09/16 VUs  0m37.0s/1m00.0s\n'
           '\n'
           'running (0m38.0s), 10/16 VUs, 6 complete and 0 interrupted '
           'iterations\n'
           'default   [  63% ] 10/16 VUs  0m38.0s/1m00.0s\n'
           '\n'
           'running (0m39.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  65% ] 10/16 VUs  0m39.0s/1m00.0s\n'
           '\n'
           'running (0m40.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  67% ] 10/16 VUs  0m40.0s/1m00.0s\n'
           '\n'
           'running (0m41.0s), 10/16 VUs, 8 complete and 0 interrupted '
           'iterations\n'
           'default   [  68% ] 10/16 VUs  0m41.0s/1m00.0s\n'
           '\n'
           'running (0m42.0s), 11/16 VUs, 9 complete and 0 interrupted '
           'iterations\n'
           'default   [  70% ] 11/16 VUs  0m42.0s/1m00.0s\n'
           '\n'
           'running (0m43.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  72% ] 11/16 VUs  0m43.0s/1m00.0s\n'
           '\n'
           'running (0m44.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  73% ] 11/16 VUs  0m44.0s/1m00.0s\n'
           '\n'
           'running (0m45.0s), 11/16 VUs, 10 complete and 0 interrupted '
           'iterations\n'
           'default   [  75% ] 11/16 VUs  0m45.0s/1m00.0s\n'
           '\n'
           'running (0m46.0s), 12/16 VUs, 11 complete and 0 interrupted '
           'iterations\n'
           'default   [  77% ] 12/16 VUs  0m46.0s/1m00.0s\n'
           '\n'
           'running (0m47.0s), 12/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  78% ] 12/16 VUs  0m47.0s/1m00.0s\n'
           '\n'
           'running (0m48.0s), 12/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  80% ] 12/16 VUs  0m48.0s/1m00.0s\n'
           '\n'
           'running (0m49.0s), 13/16 VUs, 12 complete and 0 interrupted '
           'iterations\n'
           'default   [  82% ] 13/16 VUs  0m49.0s/1m00.0s\n'
           '\n'
           'running (0m50.0s), 13/16 VUs, 13 complete and 0 interrupted '
           'iterations\n'
           'default   [  83% ] 13/16 VUs  0m50.0s/1m00.0s\n'
           '\n'
           'running (0m51.0s), 13/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  85% ] 13/16 VUs  0m51.0s/1m00.0s\n'
           '\n'
           'running (0m52.0s), 13/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  87% ] 13/16 VUs  0m52.0s/1m00.0s\n'
           '\n'
           'running (0m53.0s), 14/16 VUs, 15 complete and 0 interrupted '
           'iterations\n'
           'default   [  88% ] 14/16 VUs  0m53.0s/1m00.0s\n'
           '\n'
           'running (0m54.0s), 14/16 VUs, 17 complete and 0 interrupted '
           'iterations\n'
           'default   [  90% ] 14/16 VUs  0m54.0s/1m00.0s\n'
           '\n'
           'running (0m55.0s), 14/16 VUs, 18 complete and 0 interrupted '
           'iterations\n'
           'default   [  92% ] 14/16 VUs  0m55.0s/1m00.0s\n'
           '\n'
           'running (0m56.0s), 14/16 VUs, 18 complete and 0 interrupted '
           'iterations\n'
           'default   [  93% ] 14/16 VUs  0m56.0s/1m00.0s\n'
           '\n'
           'running (0m57.0s), 15/16 VUs, 19 complete and 0 interrupted '
           'iterations\n'
           'default   [  95% ] 15/16 VUs  0m57.0s/1m00.0s\n'
           '\n'
           'running (0m58.0s), 15/16 VUs, 20 complete and 0 interrupted '
           'iterations\n'
           'default   [  97% ] 15/16 VUs  0m58.0s/1m00.0s\n'
           '\n'
           'running (0m59.0s), 15/16 VUs, 21 complete and 0 interrupted '
           'iterations\n'
           'default   [  98% ] 15/16 VUs  0m59.0s/1m00.0s\n'
           '\n'
           'running (1m00.0s), 15/16 VUs, 21 complete and 0 interrupted '
           'iterations\n'
           'default   [ 100% ] 15/16 VUs  1m00.0s/1m00.0s\n'
           '\n'
           'running (1m01.0s), 14/16 VUs, 22 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m02.0s), 13/16 VUs, 23 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m03.0s), 12/16 VUs, 24 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m04.0s), 12/16 VUs, 24 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m05.0s), 11/16 VUs, 25 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m06.0s), 09/16 VUs, 27 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m07.0s), 08/16 VUs, 28 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m08.0s), 08/16 VUs, 28 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m09.0s), 06/16 VUs, 30 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m10.0s), 05/16 VUs, 31 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m11.0s), 04/16 VUs, 32 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m12.0s), 03/16 VUs, 33 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m13.0s), 02/16 VUs, 34 complete and 0 interrupted '
           'iterations\n'
           'default ↓ [ 100% ] 15/16 VUs  1m0s\n'
           '\n'
           'running (1m13.9s), 00/16 VUs, 36 complete and 0 interrupted '
           'iterations\n'
           'default ✓ [ 100% ] 00/16 VUs  1m0s\n'}
```

#### Action - Sleep to give time for turning on all Training pods

The *action* returned the following result:

``` javascript
{'status': 0, 'stderr': '', 'stdout': ''}
```
