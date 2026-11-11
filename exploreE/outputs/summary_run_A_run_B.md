| variant | n | success | collision | withdraw+reopen | **safe recovery** [95% Wilson] | grazing before withdraw | P50 ms |
|---|---|---|---|---|---|---|---|
| GPC-RANK | 24 | 23 | 1 | 24 | **23** [0.80, 0.99] | 0 | 51 |
| GPC-RANK|noprior | 24 | 24 | 0 | 24 | **24** [0.86, 1.00] | 0 | 51 |
| Ours 4D-WM | 24 | 24 | 0 | 24 | **24** [0.86, 1.00] | 0 | 83 |
| Ours 4D-WM|noprior | 24 | 24 | 2 | 23 | **21** [0.69, 0.96] | 2 | 83 |
| Ours tau=0.7 | 24 | 23 | 1 | 21 | **21** [0.69, 0.96] | 0 | 171 |
| Ours tau=0.7|noprior | 24 | 23 | 1 | 21 | **21** [0.69, 0.96] | 0 | 171 |
| Ours tau=0.9 | 24 | 22 | 3 | 22 | **20** [0.64, 0.93] | 0 | 92 |
| Ours tau=0.9|noprior | 24 | 23 | 1 | 22 | **22** [0.74, 0.98] | 0 | 92 |

| variant | safe relaxed | safe elastic_slow | safe elastic_fast |
|---|---|---|---|
| GPC-RANK | 8/8 | 8/8 | 7/8 |
| GPC-RANK|noprior | 8/8 | 8/8 | 8/8 |
| Ours 4D-WM | 8/8 | 8/8 | 8/8 |
| Ours 4D-WM|noprior | 6/8 | 7/8 | 8/8 |
| Ours tau=0.7 | 8/8 | 7/8 | 6/8 |
| Ours tau=0.7|noprior | 8/8 | 8/8 | 5/8 |
| Ours tau=0.9 | 8/8 | 7/8 | 5/8 |
| Ours tau=0.9|noprior | 8/8 | 7/8 | 7/8 |

| comparison (A − B), paired cells | n | Δ safe recovery | seed-cluster bootstrap 95% CI | A wins / B wins | sign test p |
|---|---|---|---|---|---|
| GPC-RANK − GPC-RANK|noprior | 24 | -0.042 | [-0.125, +0.000] | 0 / 1 | 1.000 |
| Ours 4D-WM − Ours 4D-WM|noprior | 24 | +0.125 | [+0.000, +0.250] | 3 / 0 | 0.250 |
| Ours tau=0.7 − Ours tau=0.7|noprior | 24 | +0.000 | [-0.125, +0.125] | 1 / 1 | 1.000 |
| Ours tau=0.9 − Ours tau=0.9|noprior | 24 | -0.083 | [-0.167, +0.000] | 0 / 2 | 0.500 |
| Ours tau=0.9 − GPC-RANK | 24 | -0.125 | [-0.167, -0.042] | 0 / 3 | 0.250 |
| Ours tau=0.7 − GPC-RANK | 24 | -0.083 | [-0.250, +0.000] | 0 / 2 | 0.500 |
| Ours 4D-WM − GPC-RANK | 24 | +0.042 | [+0.000, +0.125] | 1 / 0 | 1.000 |
| Ours tau=0.9|noprior − GPC-RANK|noprior | 24 | -0.083 | [-0.250, +0.000] | 0 / 2 | 0.500 |
| Ours tau=0.7|noprior − GPC-RANK|noprior | 24 | -0.125 | [-0.250, +0.000] | 0 / 3 | 0.250 |
| Ours 4D-WM|noprior − GPC-RANK|noprior | 24 | -0.125 | [-0.250, +0.000] | 0 / 3 | 0.250 |