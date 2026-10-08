"""Hand-computed checks of the metric code. Runs offline in a second; every expected value is worked out in the
comments, not computed by the code under test."""
import math

from pydantic import ValidationError

from .common import Evidence, Question, chunk_text, hits, parse_json
from .harness import cohen_kappa, generation_metrics, retrieval_metrics
from .retrieve import rrf


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-4)


def run() -> None:
    # four chunks: A[0,100) A[80,180) B[0,100) A[300,400)
    cs = [{"paper_id": "A", "start": 0, "end": 100}, {"paper_id": "A", "start": 80, "end": 180},
          {"paper_id": "B", "start": 0, "end": 100}, {"paper_id": "A", "start": 300, "end": 400}]
    ev = lambda p, s, e: Evidence(paper_id=p, start=s, end=e, quote="x")
    assert hits(cs[1], ev("A", 95, 115)) and not hits(cs[0], ev("A", 95, 115))  # 20 vs 5 of 20 chars covered
    assert not hits(cs[2], ev("A", 0, 10))  # other paper
    golds = [[ev("A", 90, 100)],                      # Q1: c0 and c1 both cover it
             [ev("A", 310, 320), ev("B", 50, 60)],    # Q2: c3 covers the first, c2 the second
             [ev("A", 95, 115)]]                      # Q3: only c1 covers >= 50%
    ranked = [[2, 1, 0], [3, 0, 2], [0, 2]]
    # Q1: first hit at rank 2 (c1); c0 at rank 3 hits the same span, no second gain. DCG 1/log2(3)=0.63093, IDCG 1
    # Q2: hits at ranks 1 and 3. DCG 1 + 1/log2(4) = 1.5, IDCG 1 + 1/log2(3) = 1.63093 -> 0.91972
    # Q3: c0 covers 5 of 20 chars, below half: no hit
    m = retrieval_metrics(ranked, golds, k_list=(1, 5), cs=cs)
    assert close(m["recall@1"], 1 / 3) and close(m["recall@5"], 2 / 3), m
    assert close(m["mrr@10"], (0.5 + 1 + 0) / 3), m
    assert close(m["ndcg@10"], (0.63093 + 0.91972 + 0) / 3), m

    def q(i, t):
        return Question(id=i, type=t, question="?", gold_answer="g", evidence=[] if t == "unanswerable" else [ev("A", 0, 5)],
                        source_papers=["A"], machine_check="test")
    qs = [q("a", "factoid"), q("b", "factoid"), q("c", "unanswerable"), q("d", "list"), q("e", "unanswerable"), q("f", "negation")]
    rec = lambda ok, ab, f, c, je=False: {"gen_ok": ok, "abstained": ab, "faithful": f, "correct": c, "judge_error": je}
    recs = {"a": rec(True, False, True, True),        # pass
            "b": rec(True, False, False, True),       # correct but unfaithful: fail
            "c": rec(True, True, None, None),         # unanswerable, abstained: pass
            "d": rec(False, False, None, None),       # generation failed: fail, still in the denominator
            "e": rec(True, False, False, False),      # unanswerable, answered: fail
            "f": rec(True, False, None, True, True)}  # judge reply unparseable: fail
    g = generation_metrics(qs, recs)
    assert close(g["GAA"], 2 / 6) and g["N"] == 6 and g["answered"] == 4, g
    assert close(g["faithfulness"], 1 / 4) and close(g["correctness"], 3 / 4), g  # a,b,e,f answered
    assert close(g["abstention_acc"], 1 / 2) and close(g["false_abstention"], 0) and g["gen_failures"] == 1, g
    assert g["judge_failures"] == 1, g

    # po = 3/4, pe = .5*.25 + .5*.75 = .5, kappa = (.75 - .5) / .5 = .5
    assert close(cohen_kappa([True, True, False, False], [True, False, False, False]), 0.5)
    # 1: 1/61 + 1/62 = .032522 · 3: 1/63 + 1/61 = .032266 · 2: 1/62
    assert rrf([1, 2, 3], [3, 1]) == [1, 3, 2]

    text = " ".join(f"w{i}" for i in range(500))
    ch = chunk_text("P", text)
    assert len(ch) == 3 and ch[1]["text"].split()[0] == "w200" and ch[1]["text"].split()[-1] == "w449", ch[1]
    assert all(text[c["start"]:c["end"]] == c["text"] for c in ch) and ch[-1]["text"].split()[-1] == "w499"

    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1} and parse_json("no json") is None and parse_json("[1]") is None
    try:
        Question.model_validate({**q("z", "factoid").model_dump(), "extra": 1})
        raise AssertionError("unknown field accepted")
    except ValidationError:
        pass
    print("selftest ok")
