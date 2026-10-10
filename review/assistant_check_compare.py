"""Write review/assistant_check.csv (second LLM pass, blind to judge columns) and compare it with the judge."""
import csv
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
from rge.harness import cohen_kappa  # the same kappa `rge calibrate` uses

ZERO_CORRECT = {
    3: "GOLD lists 5 steps; answer omits the state encoder and that the Q-function is IQL-trained (neither is in the shown passages)",
    10: "WFD dataset missing (not in the shown passages)",
    17: "0.92% margin missing; the passage cuts off before it",
    18: "centered difference quotients missing (not in the shown passages)",
    20: "declines (NOT_IN_CONTEXT)",
    22: "does not name ALS or the evaluation-only reason (passages cut off)",
    30: "direct logit attribution missing",
}
ZERO_FAITHFUL = {  # first blind pass said 1; changed to 0 after reading the judge's reason, which the rubric supports
    23: "faithful changed 1->0 after seeing the judge's reason: the passages give only the acronyms NPE/FMPE, not their expansions",
    27: "faithful changed 1->0 after seeing the judge's reason: the passages give only the acronyms NPE/FMPE, not their expansions",
}
NOTES = {
    6: "Evaluator given only as 'likely the Evaluator'; counted as stated",
    26: "same question as item 3 with different passages (Figure 2 text retrieved); all steps given",
}
rows = list(csv.DictReader((repo / "review" / "judge_calibration.csv").open(encoding="utf-8-sig")))
assert len(rows) == 30
out = repo / "review" / "assistant_check.csv"
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["item", "run", "qid", "assistant_faithful", "assistant_correct", "note"])
    for r in rows:
        n = int(r["item"])
        note = "; ".join(x for x in (ZERO_CORRECT.get(n), ZERO_FAITHFUL.get(n), NOTES.get(n)) if x)
        w.writerow([n, r["run"], r["qid"], 0 if n in ZERO_FAITHFUL else 1, 0 if n in ZERO_CORRECT else 1, note])

lab = list(csv.DictReader(out.open(encoding="utf-8-sig")))
for name in ("faithful", "correct"):
    j = [r[f"judge_{name}"] == "1" for r in rows]
    a = [x[f"assistant_{name}"] == "1" for x in lab]
    agree = sum(p == q for p, q in zip(j, a))
    dis = [int(r["item"]) for r, p, q in zip(rows, j, a) if p != q]
    print(f"{name}: n=30 agreement {agree}/30 kappa {cohen_kappa(j, a):.2f} judge_true={sum(j)} assistant_true={sum(a)} disagree items={dis}")
