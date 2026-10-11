# Faithfulness rubric v2 (measured 2026-10-11)

Same committed answers, passages, judge (claude-opus-5-5, effort low) and correctness verdicts (v1); only the faithfulness system prompt changed (rubric v2 in rge/harness.py). v2 was written after the person's labels on the 30-item sheet showed three lenient misses, so agreement on those 30 is in-sample, not a held-out test.

| run | faithfulness v1 | faithfulness v2 | GAA v1 | GAA v2 |
|---|---:|---:|---:|---:|
| R1a | 0.979 | 0.958 | 0.714 | 0.707 |
| R1b | 0.979 | 0.979 | 0.721 | 0.721 |
| R1c | 0.990 | 0.958 | 0.729 | 0.707 |
| R2 | 0.989 | 0.945 | 0.679 | 0.657 |
| R3 | 0.989 | 0.947 | 0.721 | 0.707 |
| R4 | 0.988 | 0.941 | 0.593 | 0.579 |

Per item, v1 vs v2 on 560 answered (run, question) pairs: match 0.970, v1 faithful 552, v2 faithful 535, flipped to unfaithful 14, flipped to faithful 0; v2 parse failures 3 (counted as not faithful in the GAA columns, as judge failures are in v1).

Against the person on the calibration sheet (n = 30, in-sample for v2): v1 agreement 27/30, kappa 0.53; v2 agreement 28/30, kappa 0.71 (accept at >= 0.6).

Spend: runs/<batch>/usage.json for the judge-faith-v2 batch (gitignored); total in the README.

Noise floor (v1 judged again, cache off, 560 pairs, batch msgbatch_016wGE5yUj5wxZ7Ma7FVgTeU): match 0.995, flipped to unfaithful 3, flipped to faithful 0, parse failures 0. So the 14 v1-to-v2 flips are about five times the judge's own run-to-run flips, all in one direction.
