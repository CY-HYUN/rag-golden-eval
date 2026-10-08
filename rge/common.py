"""Shared pieces: paths, pinned models and prices, the corpus and chunk readers, the golden-set schema,
and one Message Batches helper that every LLM stage goes through (budget check, cache, resume)."""
import gzip
import hashlib
import json
import os
import time
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(os.environ.get("RGE_ROOT", Path(__file__).resolve().parent.parent))
DATA, RUNS, RESULTS, REVIEW = ROOT / "data", ROOT / "runs", ROOT / "results", ROOT / "review"
CORPUS = DATA / "corpus.jsonl.gz"
GOLDEN = DATA / "golden_set_v1.jsonl"
LLM_CACHE = ROOT / "cache" / "llm.jsonl"
KEY_FILE = Path(os.environ.get("RGE_API_KEY_FILE", Path.home() / ".anthropic" / "text2cad_api.env"))

# Pinned: a model change is a new results table, never a silent swap.
GEN_MODEL = "claude-sonnet-5-5"
JUDGE_MODEL = "claude-opus-5-5"  # also drafts the golden-set questions
EMB_MODEL, EMB_REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
RERANK_MODEL, RERANK_REV = "cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"

# Message Batches API prices, USD per million tokens (input, output), read 2026-10-08 from
# https://platform.claude.com/docs/en/about-claude/pricing (Batch processing table, 50% of list price).
BATCH_PRICE = {"claude-sonnet-5-5": (1.0, 5.0), "claude-opus-5-5": (2.0, 10.0)}
BUDGET_USD = float(os.environ.get("RGE_BUDGET_USD", "60"))

CHUNK_WORDS, CHUNK_STRIDE = 250, 200  # fixed word windows, 50-word overlap
TOP_K_GEN = 5  # chunks shown to the generator; fixed a priori for every rung (not tuned)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper_id: str
    start: int = Field(ge=0)  # character span of the verbatim quote in the paper's stored text
    end: int
    quote: str


class Question(BaseModel):
    """One golden-set record. Validation is strict: no unknown fields, no defaults filled in silently."""
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["factoid", "multi_hop", "negation", "unanswerable", "list"]
    question: str
    gold_answer: str
    evidence: list[Evidence]  # empty only for unanswerable
    source_papers: list[str]
    machine_check: str  # how the record passed the automatic check (never "human-verified")


def load_golden(path: Path = GOLDEN) -> list[Question]:
    qs = [Question.model_validate_json(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for q in qs:
        if (q.type == "unanswerable") != (not q.evidence):
            raise ValueError(f"{q.id}: evidence must be empty exactly for unanswerable questions")
    return qs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def papers() -> dict[str, dict]:
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        return {p["id"]: p for p in map(json.loads, f)}


def chunk_text(pid: str, text: str) -> list[dict]:
    """Fixed word windows with overlap; each chunk keeps its character span in the paper text."""
    starts, pos = [], 0
    for w in text.split(" "):
        starts.append(pos)
        pos += len(w) + 1
    out = []
    for i, a in enumerate(range(0, max(len(starts) - CHUNK_WORDS + CHUNK_STRIDE, 1), CHUNK_STRIDE)):
        b = min(a + CHUNK_WORDS, len(starts))
        s, e = starts[a], (starts[b] - 1 if b < len(starts) else len(text))
        out.append({"id": f"{pid}#{i}", "paper_id": pid, "start": s, "end": e, "text": text[s:e]})
    return out


@cache
def chunks() -> list[dict]:
    return [c for pid, p in sorted(papers().items()) for c in chunk_text(pid, p["text"])]


def hits(chunk: dict, ev: Evidence) -> bool:
    """A retrieved chunk hits a gold evidence span when it covers at least half of that span."""
    if chunk["paper_id"] != ev.paper_id:
        return False
    overlap = min(chunk["end"], ev.end) - max(chunk["start"], ev.start)
    return overlap >= 0.5 * (ev.end - ev.start)


# ---------------------------------------------------------------- LLM calls (Message Batches only)

def client():
    import anthropic
    return anthropic.Anthropic(api_key=KEY_FILE.read_text(encoding="utf-8").strip())  # key goes to the client only


def params(model: str, system: str, user: str, max_tokens: int, effort: str) -> dict:
    p = {"model": model, "max_tokens": max_tokens, "system": system,
         "messages": [{"role": "user", "content": user}], "output_config": {"effort": effort}}
    if model == GEN_MODEL:
        p["thinking"] = {"type": "between_tools"}  # Sonnet 5.5's way to answer without a thinking pass
    return p


def key_of(p: dict) -> str:
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


def spent_usd() -> float:
    return sum(json.loads(f.read_text())["usd"] for f in RUNS.glob("*/usage.json"))


def _llm_cache() -> dict:
    if not LLM_CACHE.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, LLM_CACHE.read_text(encoding="utf-8").splitlines())}


def run_batch(reqs: dict[str, dict], tag: str, est_out: int, use_cache: bool) -> dict[str, dict]:
    """Send {custom_id: params} as one batch, wait, return {custom_id: {"text", "stop", "ok"}}.
    use_cache: answers keyed by the hash of the full request (judge and check calls); off for generation,
    whose repeats are the noise measurement. A batch with the same tag and request hash is resumed, not re-sent."""
    cached = _llm_cache() if use_cache else {}
    out = {cid: cached[key_of(p)] for cid, p in reqs.items() if key_of(p) in cached}
    todo = {cid: p for cid, p in reqs.items() if cid not in out}
    print(f"{tag}: {len(reqs)} requests, {len(out)} from cache, {len(todo)} to send")
    if not todo:
        return out
    rhash = key_of(todo)
    bid = next((m["batch"] for m in (json.loads(f.read_text()) for f in RUNS.glob("*/meta.json"))
                if m["tag"] == tag and m["reqs_hash"] == rhash), None)
    c = client()
    if bid is None:
        est = sum(len(json.dumps(p)) / 3.5 * BATCH_PRICE[p["model"]][0] + est_out * BATCH_PRICE[p["model"]][1]
                  for p in todo.values()) / 1e6
        if spent_usd() + est > BUDGET_USD:
            raise SystemExit(f"STOP: spent ${spent_usd():.2f} + estimate ${est:.2f} for {tag} > budget ${BUDGET_USD}")
        print(f"{tag}: estimated ${est:.2f}, spent so far ${spent_usd():.2f}")
        bid = c.messages.batches.create(requests=[{"custom_id": k, "params": v} for k, v in todo.items()]).id
        (RUNS / bid).mkdir(parents=True, exist_ok=True)
        (RUNS / bid / "meta.json").write_text(json.dumps({"batch": bid, "tag": tag, "reqs_hash": rhash, "n": len(todo),
                                                          "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    while (b := c.messages.batches.retrieve(bid)).processing_status != "ended":
        print(f"{tag} {bid}: {b.request_counts}", flush=True)
        time.sleep(30)
    got, usage = {}, {"in": 0, "out": 0, "usd": 0.0, "errored": 0}
    for r in c.messages.batches.results(bid):
        p = todo[r.custom_id]
        if r.result.type != "succeeded":
            usage["errored"] += 1
            got[r.custom_id] = {"text": "", "stop": r.result.type, "ok": False}
            continue
        m = r.result.message
        pin, pout = BATCH_PRICE[p["model"]]
        usage["in"] += m.usage.input_tokens
        usage["out"] += m.usage.output_tokens
        usage["usd"] += (m.usage.input_tokens * pin + m.usage.output_tokens * pout) / 1e6
        text = "".join(x.text for x in m.content if x.type == "text")
        got[r.custom_id] = {"text": text, "stop": m.stop_reason, "ok": m.stop_reason == "end_turn"}
    (RUNS / bid / "results.jsonl").write_text("".join(json.dumps({"custom_id": k, **v}) + "\n" for k, v in got.items()),
                                              encoding="utf-8")
    (RUNS / bid / "usage.json").write_text(json.dumps(usage))
    print(f"{tag} {bid}: done {usage}")
    if use_cache:
        LLM_CACHE.parent.mkdir(exist_ok=True)
        with LLM_CACHE.open("a", encoding="utf-8") as f:
            for k, v in got.items():
                if v["ok"]:  # failures are not cached, so a re-run retries them
                    f.write(json.dumps({"key": key_of(todo[k]), **v, "batch": bid}) + "\n")
    return {**out, **{k: {**v, "batch": bid} for k, v in got.items()}}


def parse_json(text: str) -> dict | None:
    """The first JSON object in a reply (code fences allowed), or None. None is a recorded failure, never a pass."""
    s = text.find("{")
    e = text.rfind("}")
    if s < 0 or e < s:
        return None
    try:
        v = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, dict) else None
