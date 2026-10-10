# rag-golden-eval

A question-answering system over recent arXiv papers, built eval-first: the golden set and the scorers came before
the pipeline, and every retrieval rung is measured with the same harness, prompt, k and models.

The question it answers: when the retriever changes, did the answers get better, and by more than run-to-run noise?

## What is measured

**Corpus.** 150 arXiv papers with primary category cs.CL or cs.LG, first submitted in July 2026 (after the
generator's stated knowledge cutoff of June 2026, so the closed-book control cannot lean on memory), licensed
CC BY 4.0 only. Metadata from arXiv's OAI-PMH interface (the Atom search API does not return the license), text
from the PDF through pymupdf, each paper cut to its first 10,000 words. The snapshot is committed
(`data/corpus.jsonl.gz`) with the SHA-256 of its uncompressed content in `data/manifest.json`. Papers were split
once, before anything else, into a dev slice (22) and an eval slice (128): `data/split.json`.

**Chunks.** Fixed windows of 250 words with a 50-word overlap. Every chunk keeps its character span in the paper.

**Golden set v1** (`data/golden_set_v1.jsonl`, schema = `Question` in `rge/common.py`, loaded strictly). Questions
were drafted by `claude-opus-5-5` from eval-slice passages, one per call, with the type mix of the design doc
(factoid, multi-hop, negation/contrastive, unanswerable, list). Each answerable question carries its gold evidence:
paper id, character span and the verbatim quote. A draft was kept only when it passed a machine check:

1. every quote is found verbatim in the passage it cites;
2. a separate call (the generator model and the harness prompt) answers the question from the cited passages only,
   and the harness's correctness judge finds that answer consistent with the gold answer;
3. unanswerable questions: shown the 8 BM25 and 8 dense top passages from the whole corpus, the same call abstains.

Every draft and the reason any was dropped is in `data/golden_drafts.jsonl`. **No question has been checked by a
person yet.** `review/human_check.csv` is the sheet for that (a ✓/✗ column per question). The SHA-256 of the
golden set was committed before the first measurement, and `measure` refuses to run if the file or its hash
differs from what is committed.

**The ladder.** Only the retrieval stage changes between rungs; the generator always sees the top 5 chunks.

| rung | retrieval |
|---|---|
| R0 | none: the same prompt with "Context passages: none" (closed-book control) |
| R1 | BM25 (`bm25s`, English stopwords, Snowball stemmer) |
| R2 | dense: `BAAI/bge-small-en-v1.5` @ `5c38ec7c`, Qdrant local mode, cosine |
| R3 | reciprocal rank fusion of R1 and R2 (k = 60, top 50 of each) |
| R4 | R3's top 20 re-scored by `cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d2` |

**Retrieval metrics** (answerable questions only; their count is in the table). A retrieved chunk hits a gold
evidence span when it covers at least half of it. recall@k: at least one hit in the top k. MRR@10: 1 / rank of the
first hit. nDCG@10: binary gain, each gold span counted once, ideal = min(#spans, 10) hits at the top.

**Generation metrics.** Generator `claude-sonnet-5-5`, judge `claude-opus-5-5`, both through the Message Batches
API. The prompt asks for `NOT_IN_CONTEXT` when the question cannot be answered; a reply starting with it is an
abstention.

- faithfulness: the judge, shown the passages, finds every claim supported. Denominator: answered questions.
  R0 shows no passages, so an R0 answer is unfaithful by definition (no judge call).
- correctness: the judge, shown the gold answer, finds the answer consistent. Denominator: answered questions.
- abstention accuracy: abstained / unanswerable questions. False abstention: abstained / answerable questions.
- **GAA (grounded answer accuracy)**: answerable and (correct and faithful), or unanswerable and abstained, divided
  by **all N questions**. A failed generation or an unparseable judge reply counts as a failure, never as a pass
  and never as an exclusion; the counts are in the table.

Judge replies are cached by the hash of the full request (`cache/llm.jsonl`), so re-running costs nothing.

## Run it

```
uv sync
uv run python -m rge report        # the results table below, from the committed results/, no API calls
```

The full pipeline, in order (needs an Anthropic API key in `%USERPROFILE%/.anthropic/text2cad_api.env`, one line,
or the path in `RGE_API_KEY_FILE`; the key is read from the file and passed to the client only):

```
uv run python -m rge selftest      # metric functions on hand-computed examples
uv run python -m rge corpus        # fetch, filter, extract, hash, split (only to rebuild the snapshot)
uv run python -m rge index         # BM25 + dense index over all chunks (79 min for the 5,354 chunks on this laptop's CPU)
uv run python -m rge golden        # draft and machine-check the golden set (cached batches are reused)
uv run python -m rge measure       # the ladder through generation and the judge
uv run python -m rge repeat        # R3 and R4 once more in a new generation batch: the repeat rows
uv run python -m rge judge-sensitivity [model ...]   # re-judge R1a-R4 with other judges: results/judge_sensitivity.md
uv run python -m rge calibrate     # judge vs human agreement, once review/judge_calibration.csv is labelled
```

Every LLM stage checks the cumulative spend in `runs/*/usage.json` plus its own estimate against a $60 budget
(`RGE_BUDGET_USD`) before it sends anything, and stops if it would go over.

## Results

Measured 2026-10-08 on golden_set_v1 (SHA-256 `39f24b7f…`, N = 140, 119 answerable, 21 unanswerable), code at
commit `c95209c`. All generation in batch `msgbatch_01JWp1o3ySwk7ZcnsnTvLFoT`; judge calls in
`msgbatch_01BgYEvXrgK5aZLk4YC2DX79` (4 identical requests were served from `msgbatch_015HoLTb1psKA8DbSAtHFpEi`
through the cache). Per-question results: `results/<run>.jsonl`; the table: `results/summary.json`. The repeat rows
were added 2026-10-10 (`rge repeat`: the same R3 and R4 requests in a new generation batch
`msgbatch_01RuoVYeoxwMn7Zfn335idkX`, judged in `msgbatch_016YEaHynHFUVjJodYwEkqhF`; 14 of the 354 judge requests
were answers identical to the first run and came from the cache): `results/R3-rep.jsonl`, `results/R4-rep.jsonl`.

**Judge not yet calibrated**: faithfulness, correctness and GAA below come from `claude-opus-5-5` with rubric v1 and
no human agreement check yet. Two cheaper judges re-scored the same answers on 2026-10-10 (`rge judge-sensitivity`,
`results/judge_sensitivity.md`); the ranking held, see "Judge sensitivity" below.

| run | recall@5 | recall@10 | MRR@10 | nDCG@10 | faithfulness | correctness | abstention acc. | false abstention | GAA | GAA vs R1 mean |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 none | - | - | - | - | - (0 answered) | - | 1.000 | 1.000 | 0.150 | -0.571 |
| R1a BM25 | 0.765 | 0.891 | 0.559 | 0.553 | 0.979 | 0.844 | 1.000 | 0.193 | 0.714 | |
| R1b BM25 | 0.765 | 0.891 | 0.559 | 0.553 | 0.979 | 0.845 | 1.000 | 0.185 | 0.721 | |
| R1c BM25 | 0.765 | 0.891 | 0.559 | 0.553 | 0.990 | 0.854 | 1.000 | 0.193 | 0.729 | |
| R2 dense | 0.689 | 0.849 | 0.468 | 0.481 | 0.989 | 0.824 | 1.000 | 0.235 | 0.679 | -0.043 |
| R3 hybrid | 0.731 | 0.916 | 0.499 | 0.520 | 0.989 | 0.853 | 1.000 | 0.202 | 0.721 | +0.000 (within noise) |
| R3 hybrid, repeat | 0.731 | 0.916 | 0.499 | 0.520 | 0.979 | 0.853 | 1.000 | 0.202 | 0.714 | -0.007 (within noise); vs R3 -0.007 |
| R4 hybrid + rerank | 0.588 | 0.807 | 0.419 | 0.442 | 0.988 | 0.741 | 1.000 | 0.286 | 0.593 | -0.129 |
| R4 hybrid + rerank, repeat | 0.588 | 0.807 | 0.419 | 0.442 | 0.963 | 0.793 | 1.000 | 0.311 | 0.593 | -0.129; vs R4 +0.000 |

Generation failures 0 and judge parse failures 0 in every run. Answered questions per run: R0 0, R1 96 / 97 / 96,
R2 91, R3 95, R4 85; repeats R3 95, R4 82.

**Noise.** The three R1 runs had identical input. GAA spread (max - min) is **0.014** (2 questions of 140);
faithfulness 0.010, correctness 0.010. A difference smaller than 0.014 GAA is reported as within noise. Three runs
give a rough floor, not a confidence interval. The R3 and R4 repeats (2026-10-10) moved GAA by -0.007 and +0.000,
inside that band, but the totals hide most of the movement: per question, 1 of 140 R3 passes and 6 of 140 R4 passes
flipped (3 gained, 3 lost; 3 abstentions changed), and only 48 (R3) and 57 (R4) of the 140 answer texts were
identical to the first run.

**Judge sensitivity** (`results/judge_sensitivity.md`, 2026-10-10). The same answers re-judged by `claude-sonnet-5-5`
and `claude-haiku-4-5-20251001` with the same rubric and prompts: the order R1 > R3 > R2 > R4 and the best rung
(BM25) are the same under all three judges. Sonnet's GAA is 0.004-0.029 below Opus's per run (per-item pass
verdicts: 94.8% exact match, Cohen's kappa 0.84; correctness kappa 0.95). Haiku's is 0.043-0.065 below (pass
87.1%, kappa 0.64), mostly by calling unfaithful what Opus accepts, plus 1-4 unparseable replies per run, which
count as failures. Kappa on faithfulness is low for both (0.22, -0.03) because Opus marks 98.6% of answers faithful,
so there is little left to agree on beyond the base rate. None of this says which judge is right; only the human
sheet can.

**Second LLM pass on the calibration sheet** (`review/assistant_check.csv`, 2026-10-10). Not a human check: Claude
Opus 5.5 in an interactive session read the 30 items of `review/judge_calibration.csv` against the rubric, with the
per-item judge columns hidden (the judge's totals, 28 faithful and 23 correct of 30, had been printed before).
Correctness is judged against the same LLM-drafted gold answers the judge used. Correctness matched the judge on 30 of 30 (7 marked incorrect by both). Faithfulness matched on
28 of 30 at first; the two disagreements (items 23 and 27, both "NPE (Neural Posterior Estimation)") were changed to
unfaithful after reading the judge's reason, because the passages give only the acronyms. Same model family as the
judge, and two labels set after seeing its verdict, so this is a consistency check, not a calibration.

**Headline.** Best GAA is BM25 at 0.721 (mean of three runs, N = 140, frozen golden set, judge not calibrated).
Hybrid ties it (0.721, within noise). Dense alone is 0.043 below, and adding the cross-encoder reranker is 0.129
below, with recall@5 falling from 0.731 to 0.588. H1 (hybrid + rerank beats BM25) is not supported on this set.

What the numbers do and do not say:

- **R0 abstained on all 140 questions.** The shared prompt says to reply NOT_IN_CONTEXT when the question cannot be
  answered reliably, and with no passages the generator always did. So the post-cutoff corpus does need retrieval
  (H2), but R0 tells nothing about how much the model knows: its GAA of 0.150 is exactly the unanswerable share.
- **Abstention accuracy is 1.000 everywhere and carries no signal**: unanswerable questions were kept only when this
  generator, shown BM25 and dense passages, abstained (see Limits).
- **Rerank hurts here; why is not yet known.** The reranker's order is not inverted (checked on a toy pair and on
  gold passages), so the drop is real ranking behaviour of an MS MARCO cross-encoder on paper text, re-scoring only
  the top 20. One candidate cause: every question names its method or system, which gives BM25 an exact-match
  anchor. Neither is tested yet; per the design doc, the per-question error analysis happens once, after this table.

Spend (Message Batches prices, read 2026-10-08 from platform.claude.com/docs/en/about-claude/pricing): 6.12 M input
and 0.34 M output tokens, **$11.92** in total ($3.54 building the golden set, $8.38 measuring). The 2026-10-10 runs
added 5.46 M input and 0.30 M output tokens, **$6.66** (judge sensitivity $3.98: Sonnet $2.87, Haiku $1.11; the R3/R4
repeat $2.68: generation $1.05, judge $1.63), from each batch's `usage` fields.

## What changed from the design doc, and why

- **No Docker.** This machine does not run Docker. Qdrant runs in `qdrant-client`'s local mode, on disk under
  `data/index/qdrant`. The Compose stack (Qdrant, FastAPI, harness) is a next step.
- **One API provider.** The only key available is Anthropic's, so the judge (`claude-opus-5-5`) is from the same
  family as the generator (`claude-sonnet-5-5`). Design doc 5.2 asks for a different family to limit
  self-preference bias; this run does not meet that.
- **No temperature 0.** These models do not accept sampling parameters, so every run uses the API default. The three
  R1 runs measure the noise that results.
- **Golden set drafted by an LLM and checked by machine**, not authored and verified by hand. The human pass is the
  sheet in `review/`.
- **150 papers, 10,000 words each.** This laptop's CPU embeds about one chunk per second (measured 2026-10-08), so
  the corpus was held at the low end of the planned 150-300 papers; the dense index then took 79 minutes.
- **k was not tuned.** No dev-slice questions exist yet; k = 5 was fixed before any measurement.

## Limits

- The judge is not calibrated. Every judge-derived number carries "judge not yet calibrated" until
  `review/judge_calibration.csv` (30 items, stratified by rung and question type) is labelled and
  `rge calibrate` shows kappa >= 0.6 (design threshold). A second LLM pass agrees with it (see "Second LLM pass"
  above), which rules out sloppy judging but not a bias shared by the model family.
- The golden set has not been read by a person. Machine checks catch quotes that do not exist and gold answers that
  the cited passages do not support; they do not catch a question that is ambiguous, trivial or badly typed.
- Unanswerable questions were vetted with BM25 and dense passages, so they are by construction ones those two
  retrievers do not answer. That favours abstention on R1-R3 in a way a third retriever would not share.
- PDF text extraction keeps equations and tables as flattened text; references are cut, appendices past 10,000 words
  are cut.
- N is about 140, so per-type numbers rest on 14-56 questions each and are low-confidence.

## Next steps

1. A person labels `review/human_check.csv` and `review/judge_calibration.csv`; then `rge calibrate`, and a
   golden_set_v2 if any question is dropped (v1 and v2 numbers never share a table).
2. Dev-slice questions (about 30) to tune k, fusion depth and the prompt, without touching the golden set.
3. Docker Compose (Qdrant, FastAPI `/ask` + `/health`, harness profile), then a CI regression gate whose thresholds
   come from the measured noise floor.
4. A judge from another model family.

## License

Code: MIT (`LICENSE`). `data/corpus.jsonl.gz` holds text extracted from 150 arXiv papers, each under CC BY 4.0 and
kept with its own arXiv id, title, authors and license URL; the text was extracted from the PDF and cut at 10,000
words. The golden set and results quote short passages from those papers.
