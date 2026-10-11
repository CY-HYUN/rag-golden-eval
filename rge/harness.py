"""The evaluation harness: retrieval metrics, generation, the judge, GAA, the results table, judge calibration.

Generation metrics follow the design doc, section 5.2. A failed generation or an unparseable judge reply is a
recorded failure that counts against the question; nothing is skipped or defaulted to a pass."""
import csv
import json
import math
import random
import subprocess

from .common import (GEN_MODEL, GOLDEN, JUDGE_MODEL, RESULTS, REVIEW, ROOT, TOP_K_GEN, Question, chunks, hits,
                     load_golden, params, parse_json, run_batch, sha256)

ABSTAIN = "NOT_IN_CONTEXT"
GEN_SYSTEM = (
    "You answer questions about recent machine-learning research papers. When context passages are given, base "
    "the answer on them. Answer in one to three sentences. If you cannot answer the question reliably, reply with "
    f"exactly {ABSTAIN} and nothing else.")

RUBRIC_VERSION = "v1"
FAITH_SYSTEM = (
    "You check whether an answer is supported by the context passages it was written from. Faithful means every "
    "factual claim in the answer is stated in, or follows directly from, the passages. A claim that is plausible but "
    "not in the passages makes the answer unfaithful. Citation markers such as [2] and wording differences do not "
    "matter. Example: passages say 'trained on 8 GPUs for 3 days'; answer 'trained on 8 A100 GPUs' is unfaithful "
    "(A100 is not stated). Reply with one JSON object only: "
    '{"faithful": true or false, "unsupported": "the first unsupported claim, or empty"}')
CORRECT_SYSTEM = (
    "You compare a candidate answer with a reference answer to the same question. Correct means the candidate states "
    "the reference's key facts and contradicts none of them; extra detail is fine unless it contradicts. If the "
    "reference lists several items, the candidate must give all of them. A candidate that declines to answer is "
    "incorrect. Example: reference 'BLEU rose from 31.2 to 33.0'; candidate 'BLEU improved by about 1.8 points' is "
    "correct; candidate 'BLEU improved' is incorrect (key number missing). Reply with one JSON object only: "
    '{"correct": true or false, "why": "one short sentence"}')

RUNS_PLAN = ["R0", "R1a", "R1b", "R1c", "R2", "R3", "R4"]  # R1 three times on identical input: the noise floor


# ---------------------------------------------------------------- prompts

def passages(ids: list[int]) -> str:
    if not ids:
        return "Context passages: none."
    cs = chunks()
    return "Context passages:\n" + "\n\n".join(f"[{n}] (arXiv {cs[i]['paper_id']}) {cs[i]['text']}" for n, i in enumerate(ids, 1))


def gen_request(question: str, ids: list[int]) -> dict:
    # between_tools: Sonnet 5.5's way to answer without a thinking pass. Judges keep their default thinking.
    return params(GEN_MODEL, GEN_SYSTEM, f"{passages(ids)}\n\nQuestion: {question}", 1024, "high") | {"thinking": {"type": "between_tools"}}


def faith_request(question: str, ids: list[int], answer: str, model: str = JUDGE_MODEL) -> dict:
    return params(model, FAITH_SYSTEM, f"{passages(ids)}\n\nQuestion: {question}\n\nAnswer: {answer}", 3000, "low")


def correct_request(question: str, gold: str, answer: str, model: str = JUDGE_MODEL) -> dict:
    return params(model, CORRECT_SYSTEM,
                  f"Question: {question}\n\nReference answer: {gold}\n\nCandidate answer: {answer}", 3000, "low")


def abstained(text: str) -> bool:
    return text.strip().startswith(ABSTAIN)


# ---------------------------------------------------------------- metrics (pure functions; see selftest)

def retrieval_metrics(ranked: list[list[int]], golds: list[list], k_list=(5, 10), cs: list[dict] | None = None) -> dict:
    """ranked[q]: chunk indexes best first; golds[q]: that question's Evidence list (answerable questions only).
    recall@k: at least one gold span hit in the top k. MRR@10: 1/rank of the first hit. nDCG@10: binary gain, each
    gold span counted once (at the first chunk that hits it), ideal = min(#gold spans, 10) hits at the top."""
    cs = cs or chunks()
    out = {f"recall@{k}": 0.0 for k in k_list} | {"mrr@10": 0.0, "ndcg@10": 0.0}
    for r, gold in zip(ranked, golds):
        seen, gains = set(), []
        for i in r[:10]:
            new = {j for j, ev in enumerate(gold) if j not in seen and hits(cs[i], ev)}
            seen |= new
            gains.append(1.0 if new else 0.0)
        first = next((n for n, g in enumerate(gains, 1) if g), None)
        for k in k_list:
            out[f"recall@{k}"] += any(gains[:k])
        out["mrr@10"] += 1.0 / first if first else 0.0
        idcg = sum(1 / math.log2(n + 1) for n in range(1, min(len(gold), 10) + 1))
        out["ndcg@10"] += sum(g / math.log2(n + 1) for n, g in enumerate(gains, 1)) / idcg
    return {k: v / len(golds) for k, v in out.items()}


def score_question(q: Question, rec: dict) -> bool:
    """GAA pass rule (design doc 2.3)."""
    if q.type == "unanswerable":
        return rec["gen_ok"] and rec["abstained"]
    return rec["gen_ok"] and not rec["abstained"] and rec["faithful"] is True and rec["correct"] is True


def generation_metrics(qs: list[Question], recs: dict[str, dict]) -> dict:
    ans = [q for q in qs if q.type != "unanswerable"]
    una = [q for q in qs if q.type == "unanswerable"]
    answered = [q for q in qs if recs[q.id]["gen_ok"] and not recs[q.id]["abstained"]]
    pct = lambda n, d: n / d if d else float("nan")
    return {
        "N": len(qs), "answered": len(answered),
        "faithfulness": pct(sum(recs[q.id]["faithful"] is True for q in answered), len(answered)),
        "correctness": pct(sum(recs[q.id]["correct"] is True for q in answered), len(answered)),
        "correct_any_grounding": pct(sum(recs[q.id]["correct"] is True for q in ans), len(ans)),
        "abstention_acc": pct(sum(recs[q.id]["gen_ok"] and recs[q.id]["abstained"] for q in una), len(una)),
        "false_abstention": pct(sum(recs[q.id]["gen_ok"] and recs[q.id]["abstained"] for q in ans), len(ans)),
        "GAA": sum(score_question(q, recs[q.id]) for q in qs) / len(qs),
        "gen_failures": sum(not recs[q.id]["gen_ok"] for q in qs),
        "judge_failures": sum(recs[q.id]["judge_error"] for q in qs),
    }


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


# ---------------------------------------------------------------- the ladder

def check_frozen() -> str:
    """Refuse to measure unless the golden set's hash file matches the file and is committed unchanged."""
    h = sha256(GOLDEN)
    rec = (ROOT / "data" / "golden_set_v1.sha256").read_text().split()[0]
    if h != rec:
        raise SystemExit(f"golden set hash {h} != recorded {rec}: edit means golden_set_v2 and a full re-run")
    tracked = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "data/golden_set_v1.jsonl",
                              "data/golden_set_v1.sha256"]).returncode == 0
    if not tracked or not subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                                          "data/golden_set_v1.jsonl", "data/golden_set_v1.sha256"],
                                         capture_output=True).returncode == 0:
        raise SystemExit("golden set and its hash must be committed before the first measurement")
    return h


def retrieve_all(qs: list[Question]) -> dict[str, dict[str, list[int]]]:
    from .retrieve import RUNGS
    path = RESULTS / "retrieval.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {r: {q.id: f(q.question, 10) for q in qs} for r, f in RUNGS.items()}
    path.write_text(json.dumps(out), newline="\n")
    return out


def judge_all(qs, runs, gens, ret, model: str = JUDGE_MODEL):
    """Faithfulness (passages shown) and correctness (gold shown) for every answered question. R0 shows no
    passages, so an R0 answer has nothing to be faithful to and is unfaithful by definition, without a call."""
    reqs = {}
    for run in runs:
        for q in qs:
            g = gens[f"{run}__{q.id}"]
            if not g["ok"] or abstained(g["text"]):
                continue
            ids = ret[run[:2]][q.id][:TOP_K_GEN] if run != "R0" else []
            if ids:
                reqs[f"{run}__{q.id}__f"] = faith_request(q.question, ids, g["text"], model)
            reqs[f"{run}__{q.id}__c"] = correct_request(q.question, q.gold_answer, g["text"], model)
    tag = f"judge-{RUBRIC_VERSION}" + ("" if model == JUDGE_MODEL else f"-{model}")
    return run_batch(reqs, tag, est_out=600, use_cache=True)


def score_run(qs, run, gens, judged, ret) -> dict[str, dict]:
    """Per-question records of one run: the answer, the two verdicts and the GAA pass."""
    recs = {}
    for q in qs:
        g = gens[f"{run}__{q.id}"]
        rec = {"id": q.id, "type": q.type, "answer": g["text"], "gen_ok": g["ok"], "gen_stop": g["stop"],
               "abstained": g["ok"] and abstained(g["text"]), "faithful": None, "correct": None, "judge_error": False,
               "retrieved": ret[run[:2]][q.id] if run != "R0" else []}
        if rec["gen_ok"] and not rec["abstained"]:
            fr = judged[f"{run}__{q.id}__f"] if run != "R0" else None
            cr = judged[f"{run}__{q.id}__c"]
            fv = parse_json(fr["text"]) if fr else {"faithful": False}  # R0: no passages, see judge_all
            cv = parse_json(cr["text"])
            rec["faithful"] = fv.get("faithful") if fv else None
            rec["correct"] = cv.get("correct") if cv else None
            rec["judge_error"] = not isinstance(rec["faithful"], bool) or not isinstance(rec["correct"], bool)
            rec["judge_why"] = [(fv or {}).get("unsupported"), (cv or {}).get("why")]
        rec["pass"] = score_question(q, rec)
        recs[q.id] = rec
    return recs


def measure(runs: list[str] = RUNS_PLAN, gen_tag: str = "generate-v1", suffix: str = "") -> None:
    """The ladder. With a suffix: a repeat of some runs (new generation batch), written next to the originals as
    results/<run><suffix>.jsonl and added to summary.json; the judge cache serves only identical answers."""
    golden_hash = check_frozen()
    qs = load_golden()
    RESULTS.mkdir(exist_ok=True)
    ret = retrieve_all(qs)
    gen_reqs = {f"{run}__{q.id}": gen_request(q.question, ret[run[:2]][q.id][:TOP_K_GEN] if run != "R0" else [])
                for run in runs for q in qs}
    gens = run_batch(gen_reqs, gen_tag, est_out=200, use_cache=False)
    judged = judge_all(qs, runs, gens, ret)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    path = RESULTS / "summary.json"
    summary = json.loads(path.read_text()) if suffix else {"golden_sha256": golden_hash, "commit": commit,
        "gen_model": GEN_MODEL, "judge_model": JUDGE_MODEL, "rubric": RUBRIC_VERSION, "top_k_gen": TOP_K_GEN, "runs": {}}
    if suffix:
        summary.setdefault("repeats", {})[suffix] = {"runs": runs, "gen_tag": gen_tag, "commit": commit}
    answerable = [q for q in qs if q.type != "unanswerable"]
    summary["n_answerable"] = len(answerable)
    for run in runs:
        recs = score_run(qs, run, gens, judged, ret)
        (RESULTS / f"{run}{suffix}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs.values()),
                                                      encoding="utf-8", newline="\n")
        m = generation_metrics(qs, recs)
        if run != "R0":
            m |= retrieval_metrics([ret[run[:2]][q.id] for q in answerable], [q.evidence for q in answerable])
        m["batches"] = sorted({gens[f"{run}__{q.id}"].get("batch", "") for q in qs} - {""})
        summary["runs"][run + suffix] = m
    summary["judge_batches"] = sorted(set(summary.get("judge_batches", [])) | {v.get("batch", "") for v in judged.values()} - {""})
    path.write_text(json.dumps(summary, indent=1), newline="\n")
    calibration_sheet(qs, ret)
    report()


# ---------------------------------------------------------------- judge sensitivity

JUDGE_ALT = ["claude-sonnet-5-5", "claude-haiku-4-5-20251001"]


def judge_sensitivity(models: list[str] = JUDGE_ALT) -> None:
    """Re-judge the committed answers of R1a-R4 with other judge models (same rubric, same effort where the model
    takes it), then: GAA per judge and rung, agreement with the Opus judge on the per-item verdicts (exact match
    and Cohen's kappa), and whether the best rung changes. R0 answered nothing, so it is the same under any judge."""
    qs = load_golden()
    s = json.loads((RESULTS / "summary.json").read_text())
    runs = [r for r in RUNS_PLAN if r != "R0"]
    ret = retrieve_all(qs)
    base = {run: {r["id"]: r for r in map(json.loads, (RESULTS / f"{run}.jsonl").read_text(encoding="utf-8").splitlines())}
            for run in runs}
    gens = {f"{run}__{qid}": {"ok": r["gen_ok"], "text": r["answer"], "stop": r["gen_stop"]}
            for run, rs in base.items() for qid, r in rs.items()}
    verdicts, metrics, batches = {JUDGE_MODEL: base}, {JUDGE_MODEL: {run: s["runs"][run] for run in runs}}, {}
    for model in models:
        judged = judge_all(qs, runs, gens, ret, model)
        batches[model] = sorted({v.get("batch", "") for v in judged.values()} - {""})
        verdicts[model] = {run: score_run(qs, run, gens, judged, ret) for run in runs}
        metrics[model] = {run: generation_metrics(qs, verdicts[model][run]) for run in runs}
    answered = [(run, qid) for run in runs for qid, r in base[run].items() if r["gen_ok"] and not r["abstained"]]
    agree = {}
    for model in models:
        for kind in ("faithful", "correct", "pass"):
            a = [verdicts[JUDGE_MODEL][run][qid][kind] is True for run, qid in answered]
            b = [verdicts[model][run][qid][kind] is True for run, qid in answered]
            agree[(model, kind)] = {"n": len(a), "match": sum(x == y for x, y in zip(a, b)) / len(a),
                                    "kappa": cohen_kappa(a, b), "true_opus": sum(a) / len(a), "true_alt": sum(b) / len(a),
                                    "only_opus": sum(x and not y for x, y in zip(a, b)), "only_alt": sum(y and not x for x, y in zip(a, b))}
    rungs = {"R1": ("R1a", "R1b", "R1c"), "R2": ("R2",), "R3": ("R3",), "R4": ("R4",)}
    gaa = {m: {r: sum(metrics[m][x]["GAA"] for x in xs) / len(xs) for r, xs in rungs.items()} for m in metrics}
    order = {m: sorted(gaa[m], key=lambda r: -gaa[m][r]) for m in gaa}
    short = lambda m: m.split("-")[1]
    lines = [f"# Judge sensitivity (measured 2026-10-10)", "",
             f"Same answers (results/R*.jsonl from the 2026-10-08 measurement), same rubric {RUBRIC_VERSION} and prompts, "
             f"judged again by {', '.join(models)}. Effort `low` where the model takes it (Haiku 4.5 rejects the effort "
             f"parameter); thinking left at each model's default (Opus and Sonnet adaptive, Haiku none). "
             f"R0 answered nothing, so its GAA (0.150) is the same under any judge and is left out. "
             f"Verdicts cached by request hash (model included) in cache/llm.jsonl.", "",
             "## GAA per judge", "", "| run | " + " | ".join(short(m) for m in metrics) + " |", "|---|" + "---|" * len(metrics)]
    for run in runs:
        lines.append(f"| {run} | " + " | ".join(f"{metrics[m][run]['GAA']:.3f}" for m in metrics) + " |")
    lines += ["", "Faithfulness / correctness over answered questions, and judge parse failures:", "",
              "| run | " + " | ".join(short(m) for m in metrics) + " |", "|---|" + "---|" * len(metrics)]
    for run in runs:
        lines.append(f"| {run} | " + " | ".join(f"{metrics[m][run]['faithfulness']:.3f} / {metrics[m][run]['correctness']:.3f} "
                                              f"({metrics[m][run]['judge_failures']} fail)" for m in metrics) + " |")
    lines += ["", "## Agreement with the Opus judge", "",
              f"Per-item verdicts on the {len(answered)} answered (run, question) pairs pooled over R1a-R4. A judge failure "
              "counts as false, as in GAA. `true` = share of items the judge marks faithful / correct / passing; the last "
              "two columns split the disagreements by direction. Kappa on faithfulness is dominated by the base rate "
              "(Opus marks almost every answer faithful, so chance agreement is already high).", "",
              "| judge | verdict | n | exact match | Cohen's kappa | true (Opus) | true (this judge) | Opus true, this false | this true, Opus false |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (model, kind), v in agree.items():
        lines.append(f"| {short(model)} | {kind} | {v['n']} | {v['match']:.3f} | {v['kappa']:.3f} | {v['true_opus']:.3f} | "
                     f"{v['true_alt']:.3f} | {v['only_opus']} | {v['only_alt']} |")
    lines += ["", "## Ranking of the rungs (GAA, R1 = mean of the three runs)", "",
              "| judge | order | best | R1 | R2 | R3 | R4 |", "|---|---|---|---:|---:|---:|---:|"]
    for m in gaa:
        lines.append(f"| {short(m)} | {' > '.join(order[m])} | {order[m][0]} | " + " | ".join(f"{gaa[m][r]:.3f}" for r in rungs) + " |")
    lines += ["", f"Batches: " + "; ".join(f"{short(m)} {', '.join(b)}" for m, b in batches.items()) + ".",
              "Spend per batch: runs/<batch>/usage.json (gitignored); the totals are in the README."]
    (RESULTS / "judge_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with (RESULTS / "judge_sensitivity.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for run, qid in answered:
            f.write(json.dumps({"run": run, "id": qid, "type": base[run][qid]["type"]} | {
                short(m): {k: verdicts[m][run][qid][k] for k in ("faithful", "correct", "pass")} for m in verdicts}) + "\n")
    print("\n".join(lines))


def calibration_sheet(qs, ret) -> None:
    """30 judged answers, stratified by run and question type (seeded), for CY to label by hand."""
    path = REVIEW / "judge_calibration.csv"
    if path.exists():
        return  # never overwrite labels
    byq = {q.id: q for q in qs}
    pool = {}
    for run in RUNS_PLAN[1:]:  # R0 faithfulness is a rule, not a judge call
        for line in (RESULTS / f"{run}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["faithful"] is not None and not r["judge_error"]:
                pool.setdefault((run, r["type"]), []).append(r | {"run": run})
    rng, picked = random.Random(20261008), []
    strata = sorted(pool)
    rng.shuffle(strata)
    while len(picked) < 30 and any(pool.values()):
        for s in strata:
            if pool[s] and len(picked) < 30:
                picked.append(pool[s].pop(rng.randrange(len(pool[s]))))
    REVIEW.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["item", "run", "qid", "type", "question", "passages_shown", "answer", "gold_answer",
                    "judge_faithful", "judge_correct", "human_faithful (1/0)", "human_correct (1/0)", "note"])
        for n, r in enumerate(picked, 1):
            q = byq[r["id"]]
            ids = r["retrieved"][:TOP_K_GEN]
            w.writerow([n, r["run"], q.id, q.type, q.question, passages(ids), r["answer"], q.gold_answer,
                        int(r["faithful"]), int(r["correct"]), "", "", ""])
    print(f"wrote {path} ({len(picked)} items)")


def calibrate() -> None:
    rows = list(csv.DictReader((REVIEW / "judge_calibration.csv").open(encoding="utf-8-sig")))
    for name in ("faithful", "correct"):
        lab = [(r[f"judge_{name}"] == "1", r[f"human_{name} (1/0)"].strip() == "1") for r in rows
               if r[f"human_{name} (1/0)"].strip() in ("0", "1")]
        if not lab:
            print(f"{name}: no human labels yet")
            continue
        j, h = zip(*lab)
        agree = sum(a == b for a, b in lab) / len(lab)
        print(f"{name}: n={len(lab)} agreement {agree:.1%} kappa {cohen_kappa(list(j), list(h)):.2f} (accept at >= 0.6)")


# ---------------------------------------------------------------- report

def report() -> None:
    s = json.loads((RESULTS / "summary.json").read_text())
    runs = s["runs"]
    noise = {k: max(runs[r][k] for r in ("R1a", "R1b", "R1c")) - min(runs[r][k] for r in ("R1a", "R1b", "R1c"))
             for k in ("GAA", "faithfulness", "correctness", "abstention_acc")}
    r1 = {k: sum(runs[r][k] for r in ("R1a", "R1b", "R1c")) / 3 for k in noise}
    cols = ["recall@5", "recall@10", "mrr@10", "ndcg@10", "faithfulness", "correctness", "abstention_acc",
            "false_abstention", "GAA", "answered", "gen_failures", "judge_failures"]
    print(f"golden {s['golden_sha256'][:12]} · N={runs['R0']['N']} (retrieval metrics over the {s['n_answerable']} answerable) · gen {s['gen_model']} · judge {s['judge_model']} "
          f"(vs one person on 30 items: judge correctness kappa 0.91, faithfulness kappa 0.53) · commit {s['commit']}")
    print("| run | " + " | ".join(cols) + " | GAA vs R1 mean |")
    print("|" + "---|" * (len(cols) + 2))
    for r in [k for p in RUNS_PLAN for k in [p] + sorted(x for x in runs if x.startswith(p + "-"))]:  # repeats after their original
        m = runs[r]
        cells = [("-" if c not in m else f"{m[c]:.3f}" if isinstance(m[c], float) else str(m[c])) for c in cols]
        d = m["GAA"] - r1["GAA"]
        tag = "" if r.startswith("R1") else f"{d:+.3f}" + (" (within noise)" if abs(d) <= noise["GAA"] else "")
        if "-" in r:
            tag += f"; vs {r.split('-')[0]} {m['GAA'] - runs[r.split('-')[0]]['GAA']:+.3f}"
        print(f"| {r} | " + " | ".join(cells) + f" | {tag} |")
    print("R1 noise (max-min over 3 runs):", {k: round(v, 3) for k, v in noise.items()})
    print("generation batches:", sorted({b for m in runs.values() for b in m["batches"]}), "judge batches:", s["judge_batches"])
