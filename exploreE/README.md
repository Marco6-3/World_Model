# exploreE: recovery-prior ablation (status 2026-11-10 23:16)

## What this tests
Does the hand-written emergency noise prior (bag_learning.py L295-301) explain the C.1 safe-recovery numbers?
`explore_decide.py` is a copy of `bag_learning.decide` plus a `|noprior` switch. With the prior on it is
bit-identical to the original (`check_equivalence.py`: 72/72 checks, max diff 0). Frozen C.1 checkpoints are
loaded read-only. Only perturbed (collar-slip) episodes are run.

## Done
- run_A (seeds 3000, 3001) and run_B (3002, 3003): 4 methods x prior on/off x 3 modes x 2 orders, 192 episodes, complete.
- run_C (prior off on C.1 seeds 2000-2002): 32/72 episodes, **interrupted**.

## Results on new seeds 3000-3003 (n=24 per variant): `outputs/summary_run_A_run_B.md`
| variant | safe recovery |
|---|---|
| GPC-RANK | 23/24 |
| GPC-RANK, no prior | **24/24** |
| Ours 4D-WM | 24/24 |
| Ours 4D-WM, no prior | 21/24 |
| Ours tau=0.7 | 21/24 (no prior: 21/24) |
| Ours tau=0.9 | 20/24 (no prior: 22/24) |

Removing the prior does **not** reduce GPC-RANK. So the hypothesis "the prior gives RANK its C.1 score" is not supported.
The C.0 to C.1 jump comes from environment and data changes that apply to every method (bag_scene.py diff):
- forced finger opening during slip (`action[6]=.038`)
- occupancy band changed from .018 to .045
- a new withdraw demonstration (x, 0, .245)
- a fixed item sequence

BC, which never uses the prior, also rose from 5/18 to 15/18 on withdraw/reopen.
Several Ours "failures" are success with no collision but no withdraw (strict-metric artefact; worth reporting
success-and-no-collision separately).

## Resume
```
C:\isaacsim_env\Scripts\python.exe exploreE\explore_eval.py --out exploreE\outputs\run_C --seeds 2000 2001 2002 --priors off
python exploreE\analyze.py exploreE\outputs\run_A exploreE\outputs\run_B exploreE\outputs\run_C C1
```
The runner skips finished episodes (episodes.jsonl journal).
