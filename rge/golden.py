"""Golden set v1: drafted by an LLM from eval-slice papers, then checked by machine. Nothing here is human-verified;
review/human_check.csv is the sheet for that pass.

Checks, in order (a draft failing any one is dropped, with the reason kept in data/golden_drafts.jsonl):
1. the JSON parses and has the fields its type needs;
2. every evidence quote is found verbatim (whitespace and case aside) in the passage it cites;
3. answerable: a separate call (the generator model, the harness prompt) answers from the cited passages only,
   and the harness's correctness judge finds that answer consistent with the gold answer;
   unanswerable: the same call, shown the 8 BM25 and 8 dense top passages from the whole corpus, abstains."""
import csv
import json
import random
import re

from .common import DATA, GOLDEN, JUDGE_MODEL, REVIEW, Evidence, Question, chunks, load_golden, papers, params, parse_json, run_batch, sha256

SEED = 20261008
# drafted per type (about 1.45x the kept quota, since checks drop drafts) and kept per type (design doc 3.1)
DRAFT = {"factoid": 80, "multi_hop": 42, "negation": 32, "unanswerable": 34, "list": 22}
KEEP = {"factoid": 56, "multi_hop": 28, "negation": 21, "unanswerable": 21, "list": 14}

TYPE_RULES = {
    "factoid": "A single fact that one passage states. Cite exactly one quote.",
    "multi_hop": "Answering needs two facts from two different passages (both needed, neither alone suffices). "
                 "Cite one quote from each of the two passages.",
    "negation": "A contrastive or negated question, such as which of several things the work did NOT do, or how X "
                "differs from Y, answerable from the passages. Cite a quote for every fact the answer relies on.",
    "list": "The answer enumerates every member of a set the passages state (for example all datasets or baselines "
            "used), 2 to 8 items. Cite quotes that together contain the whole list.",
    "unanswerable": "A specific question a reader of this work would plausibly ask (a number, setting, dataset or "
                    "comparison) that the passages and the abstract do not answer and that the paper is unlikely to "
                    "answer elsewhere. Set answer to NOT_IN_CONTEXT, evidence to [], and say in why_absent what is "
                    "missing.",
}
DRAFT_SYSTEM = (
    "You write one evaluation question for a retrieval system over 200 recent machine-learning papers. Rules: "
    "the question must identify its paper among the 200 by naming the method, model, benchmark or system (never "
    "'this paper', 'the authors' or 'the proposed method'); paraphrase instead of copying the passage wording; the "
    "gold answer is short (a phrase to two sentences) and fully supported by the cited quotes; each quote is copied "
    "character for character from one passage, 8 to 50 words. Reply with one JSON object only: "
    '{"question": str, "answer": str, "evidence": [{"passage": int, "quote": str}], "why_absent": str}')


def good_chunks(pid: str) -> list[dict]:
    """Prose chunks of a paper: mostly letters (skips tables and equations) and a full window long."""
    cs = [c for c in chunks() if c["paper_id"] == pid]
    return [c for c in cs if len(c["text"].split()) >= 200
            and sum(ch.isalpha() for ch in c["text"]) / len(c["text"]) > 0.7]


def partner(pid: str, pool: list[str]) -> str:
    """The eval paper whose title+abstract shares the most words with this one (cross-paper questions)."""
    words = lambda p: set(re.findall(r"[a-z]{4,}", (papers()[p]["title"] + " " + papers()[p]["abstract"]).lower()))
    w = words(pid)
    return max((p for p in pool if p != pid), key=lambda p: (len(w & words(p)) / len(w | words(p)), p))


def plan() -> list[dict]:
    """Seeded draft plan: each item names its type and the passages shown to the drafter."""
    rng = random.Random(SEED)
    pool = json.loads((DATA / "split.json").read_text())["eval"]
    order = rng.sample(pool, len(pool))
    items, n = [], 0
    for qtype, count in DRAFT.items():
        for k in range(count):
            pid = order[n % len(order)]
            n += 1
            gc = good_chunks(pid)
            if len(gc) < 3:
                continue
            shown = rng.sample(gc, 3)
            if qtype in ("multi_hop", "negation") and k % 2:  # half of these span two papers
                other = good_chunks(partner(pid, pool))
                if len(other) >= 2:
                    shown = rng.sample(gc, 2) + rng.sample(other, 2)
            items.append({"draft": f"d{len(items):03d}", "type": qtype, "chunks": [c["id"] for c in shown]})
    return items


def draft_prompt(item: dict) -> str:
    by = {c["id"]: c for c in chunks()}
    pids = sorted({by[c]["paper_id"] for c in item["chunks"]})
    head = "\n".join(f"Paper arXiv {p}: {papers()[p]['title']}. Abstract: {papers()[p]['abstract']}" for p in pids)
    body = "\n\n".join(f"Passage {n} (arXiv {by[c]['paper_id']}): {by[c]['text']}" for n, c in enumerate(item["chunks"], 1))
    return f"{head}\n\n{body}\n\nQuestion type: {item['type']}. {TYPE_RULES[item['type']]}"


def norm(s: str) -> str:
    return " ".join(s.split()).lower()


def locate(item: dict, d: dict) -> tuple[list[Evidence] | None, str]:
    """Map each quote to its character span in the paper text, or say why not."""
    by = {c["id"]: c for c in chunks()}
    evs = []
    for e in d.get("evidence") or []:
        if not isinstance(e, dict) or not isinstance(e.get("passage"), int) or not 1 <= e["passage"] <= len(item["chunks"]):
            return None, f"bad evidence entry {e!r}"[:200]
        c = by[item["chunks"][e["passage"] - 1]]
        q = str(e.get("quote", ""))
        low = c["text"].lower()
        if len(low) != len(c["text"]):  # lower() changed a length (rare Unicode): match case-sensitively instead
            low = c["text"]
        at = low.find(norm(q) if low is not c["text"] else " ".join(q.split()))
        if not q.strip() or at < 0:
            return None, f"quote not found verbatim in passage {e['passage']}"
        s, n = c["start"] + at, len(" ".join(q.split()))  # chunk text is single-spaced, so offsets carry over
        evs.append(Evidence(paper_id=c["paper_id"], start=s, end=s + n, quote=papers()[c["paper_id"]]["text"][s:s + n]))
    return evs, ""


def shape_problem(qtype: str, d: dict | None, evs: list[Evidence] | None) -> str:
    if d is None:
        return "reply is not a JSON object"
    if not isinstance(d.get("question"), str) or not isinstance(d.get("answer"), str) or not d["question"].strip():
        return "question or answer missing"
    if re.search(r"\b(this|the) (paper|work|study)\b|\bthe authors\b|\bthe proposed\b", d["question"], re.I):
        return "question does not name its paper"
    if qtype == "unanswerable":
        return "" if d["answer"].strip().startswith("NOT_IN_CONTEXT") and not d.get("evidence") else "unanswerable with an answer or evidence"
    if not evs:
        return "no evidence"
    if qtype == "multi_hop" and len({e["passage"] for e in d["evidence"]}) < 2:
        return "multi_hop citing fewer than two passages"
    return ""


def draft_all(items: list[dict]) -> dict[str, dict]:
    return run_batch({it["draft"]: params(JUDGE_MODEL, DRAFT_SYSTEM, draft_prompt(it), 8000, "medium") for it in items},
                     "golden-draft-v1", est_out=2500, use_cache=True)


def build(only_answerable: bool = False) -> None:
    """only_answerable: run the answerable checks alone (they need no dense index); the cache makes the full run reuse them."""
    from .harness import correct_request, gen_request, abstained
    from .retrieve import bm25, dense
    items = plan()
    drafts = draft_all(items)
    idx = {c["id"]: n for n, c in enumerate(chunks())}
    for it in items:
        r = drafts[it["draft"]]
        d = parse_json(r["text"]) if r["ok"] else None
        evs, why = locate(it, d) if d else (None, "")
        it.update(reply=d, draft_batch=r.get("batch"), evidence=[e.model_dump() for e in evs or []],
                  drop=why or shape_problem(it["type"], d, evs) or ("" if r["ok"] else f"draft call {r['stop']}"))
    live = [it for it in items if not it["drop"] and not (only_answerable and it["type"] == "unanswerable")]
    check = {}
    for it in live:
        q = it["reply"]["question"]
        if it["type"] == "unanswerable":
            ctx = list(dict.fromkeys(bm25(q, 8) + dense(q, 8)))
            it["checked_against"] = [chunks()[i]["id"] for i in ctx]
        else:
            ctx = [idx[c] for c in dict.fromkeys(it["chunks"][e["passage"] - 1] for e in it["reply"]["evidence"])]
        check[it["draft"]] = gen_request(q, ctx)
    ans = run_batch(check, "golden-reanswer-v1", est_out=200, use_cache=True)
    judge = {}
    for it in live:
        a = ans[it["draft"]]
        it["reanswer"] = a["text"]
        if not a["ok"]:
            it["drop"] = f"re-answer call {a['stop']}"
        elif it["type"] == "unanswerable":
            it["drop"] = "" if abstained(a["text"]) else "answered from corpus-wide top passages"
        elif abstained(a["text"]):
            it["drop"] = "re-answer abstained on its own evidence"
        else:
            judge[it["draft"]] = correct_request(it["reply"]["question"], it["reply"]["answer"], a["text"])
    verdict = run_batch(judge, "golden-check-judge-v1", est_out=600, use_cache=True)
    for it in live:
        if it["draft"] in verdict:
            v = parse_json(verdict[it["draft"]]["text"]) if verdict[it["draft"]]["ok"] else None
            it["check_judge"] = v
            it["drop"] = "" if v and v.get("correct") is True else "re-answer judged inconsistent with gold" if v else "check judge unparseable"
            it["check_batches"] = [ans[it["draft"]].get("batch"), verdict[it["draft"]].get("batch")]
        elif not it["drop"]:
            it["check_batches"] = [ans[it["draft"]].get("batch")]
    if not only_answerable:
        write(items)


def write(items: list[dict]) -> None:
    (DATA / "golden_drafts.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    kept, n = [], 0
    for qtype, quota in KEEP.items():
        for it in [i for i in items if i["type"] == qtype and not i["drop"]][:quota]:
            n += 1
            how = ("abstained when shown the corpus-wide BM25+dense top-8 passages" if qtype == "unanswerable"
                   else "quotes found verbatim; blind re-answer from cited passages judged consistent with gold")
            kept.append(Question(id=f"g{n:03d}", type=qtype, question=it["reply"]["question"], gold_answer=gold_of(it),
                                 evidence=[Evidence(**e) for e in it["evidence"]],
                                 source_papers=sorted({chunks_paper(c) for c in it["chunks"]}),
                                 machine_check=f"{how} (draft {it['draft']}, batches {', '.join(filter(None, it['check_batches']))})"))
    GOLDEN.write_text("".join(q.model_dump_json() + "\n" for q in kept), encoding="utf-8")
    load_golden()  # the file must load under the strict schema before it is hashed
    h = sha256(GOLDEN)
    (DATA / "golden_set_v1.sha256").write_text(f"{h}  golden_set_v1.jsonl\n")
    REVIEW.mkdir(exist_ok=True)
    with (REVIEW / "human_check.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "gold_answer", "evidence (paper: quote)", "source_papers", "machine_check",
                    "human_ok (✓/✗)", "human_note"])
        for q in kept:
            w.writerow([q.id, q.type, q.question, q.gold_answer, " | ".join(f"{e.paper_id}: \"{e.quote}\"" for e in q.evidence),
                        " ".join(q.source_papers), q.machine_check, "", ""])
    counts = {t: sum(q.type == t for q in kept) for t in KEEP}
    drops = {}
    for it in items:
        if it["drop"]:
            k = it["drop"].split(" in passage")[0][:60]
            drops[k] = drops.get(k, 0) + 1
    print(f"golden_set_v1: {len(kept)} questions {counts}, drafts {len(items)}, dropped {drops}, sha256 {h}")


def chunks_paper(cid: str) -> str:
    return cid.split("#")[0]


def gold_of(it: dict) -> str:
    a = it["reply"]["answer"].strip()
    return f"NOT_IN_CONTEXT: {it['reply'].get('why_absent') or 'not in the corpus'}" if it["type"] == "unanswerable" else a
