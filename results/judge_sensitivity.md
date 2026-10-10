# Judge sensitivity (measured 2026-10-10)

Same answers (results/R*.jsonl from the 2026-10-08 measurement), same rubric v1 and prompts, judged again by claude-sonnet-5-5, claude-haiku-4-5-20251001. Effort `low` where the model takes it (Haiku 4.5 rejects the effort parameter); thinking left at each model's default (Opus and Sonnet adaptive, Haiku none). R0 answered nothing, so its GAA (0.150) is the same under any judge and is left out. Verdicts cached by request hash (model included) in cache/llm.jsonl.

## GAA per judge

| run | opus | sonnet | haiku |
|---|---|---|---|
| R1a | 0.714 | 0.707 | 0.650 |
| R1b | 0.721 | 0.707 | 0.679 |
| R1c | 0.729 | 0.736 | 0.664 |
| R2 | 0.679 | 0.650 | 0.614 |
| R3 | 0.721 | 0.700 | 0.657 |
| R4 | 0.593 | 0.579 | 0.550 |

Faithfulness / correctness over answered questions, and judge parse failures:

| run | opus | sonnet | haiku |
|---|---|---|---|
| R1a | 0.979 / 0.844 (0 fail) | 0.948 / 0.854 (2 fail) | 0.906 / 0.781 (4 fail) |
| R1b | 0.979 / 0.845 (0 fail) | 0.948 / 0.856 (4 fail) | 0.928 / 0.784 (2 fail) |
| R1c | 0.990 / 0.854 (0 fail) | 0.990 / 0.865 (1 fail) | 0.938 / 0.771 (4 fail) |
| R2 | 0.989 / 0.824 (0 fail) | 0.923 / 0.835 (3 fail) | 0.835 / 0.791 (2 fail) |
| R3 | 0.989 / 0.853 (0 fail) | 0.937 / 0.863 (3 fail) | 0.916 / 0.800 (2 fail) |
| R4 | 0.988 / 0.741 (0 fail) | 0.976 / 0.729 (0 fail) | 0.894 / 0.694 (1 fail) |

## Agreement with the Opus judge

Per-item verdicts on the 560 answered (run, question) pairs pooled over R1a-R4. A judge failure counts as false, as in GAA. `true` = share of items the judge marks faithful / correct / passing; the last two columns split the disagreements by direction. Kappa on faithfulness is dominated by the base rate (Opus marks almost every answer faithful, so chance agreement is already high).

| judge | verdict | n | exact match | Cohen's kappa | true (Opus) | true (this judge) | Opus true, this false | this true, Opus false |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sonnet | faithful | 560 | 0.954 | 0.218 | 0.986 | 0.954 | 22 | 4 |
| sonnet | correct | 560 | 0.986 | 0.949 | 0.829 | 0.836 | 2 | 6 |
| sonnet | pass | 560 | 0.948 | 0.835 | 0.814 | 0.795 | 20 | 9 |
| haiku | faithful | 560 | 0.889 | -0.026 | 0.986 | 0.904 | 54 | 8 |
| haiku | correct | 560 | 0.921 | 0.756 | 0.829 | 0.771 | 38 | 6 |
| haiku | pass | 560 | 0.871 | 0.639 | 0.814 | 0.729 | 60 | 12 |

## Ranking of the rungs (GAA, R1 = mean of the three runs)

| judge | order | best | R1 | R2 | R3 | R4 |
|---|---|---|---:|---:|---:|---:|
| opus | R1 > R3 > R2 > R4 | R1 | 0.721 | 0.679 | 0.721 | 0.593 |
| sonnet | R1 > R3 > R2 > R4 | R1 | 0.717 | 0.650 | 0.700 | 0.579 |
| haiku | R1 > R3 > R2 > R4 | R1 | 0.664 | 0.614 | 0.657 | 0.550 |

Batches: sonnet msgbatch_01BpUfStwRTRoEoB3MSHPhjb; haiku msgbatch_01NWWHogMbbx2TeKcsJT3A3a.
Spend per batch: runs/<batch>/usage.json (gitignored); the totals are in the README.
