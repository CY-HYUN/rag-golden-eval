"""Blind copy of the judge calibration sheet for a person, and the merge back.

    python review/blind_sheet.py make    # writes review/human_label_blind.csv: no judge columns
    python review/blind_sheet.py merge   # copies its 1/0 labels into review/judge_calibration.csv
    python review/blind_sheet.py --selftest

Label each row by the judge's own rubric (rge/harness.py, FAITH_SYSTEM and CORRECT_SYSTEM, v1):
  human_faithful 1 = every factual claim in `answer` is stated in, or follows directly from, `passages_shown`.
  human_correct  1 = `answer` states the key facts of `gold_answer` and contradicts none; all items of a list;
                     a declined answer is 0.
Leave a cell empty to skip it. Then `uv run python -m rge calibrate` prints agreement and kappa.
"""
import csv
import sys
import tempfile
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
SHEET = "judge_calibration.csv"
BLIND = "human_label_blind.csv"
SHOWN = ["item", "type", "question", "passages_shown", "answer", "gold_answer"]
LABELS = {"human_faithful": "human_faithful (1/0)", "human_correct": "human_correct (1/0)"}


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def make(review: Path) -> int:
    out = review / BLIND
    if out.exists():
        raise SystemExit(f"{out} exists; delete it first so labels are never overwritten")
    rows = read(review / SHEET)
    blind = [{**{c: r[c] for c in SHOWN}, **{k: r[v] for k, v in LABELS.items()}, "note": r["note"]} for r in rows]
    write(out, blind, SHOWN + list(LABELS) + ["note"])
    return len(blind)


def merge(review: Path) -> int:
    sheet = read(review / SHEET)
    blind = {r["item"]: r for r in read(review / BLIND)}
    if set(blind) != {r["item"] for r in sheet}:
        raise SystemExit("item ids differ between the blind copy and the sheet")
    n = 0
    for r in sheet:
        b = blind[r["item"]]
        if b["question"] != r["question"]:
            raise SystemExit(f"item {r['item']}: question differs; the blind copy is from another sheet")
        for k, v in LABELS.items():
            val = b[k].strip()
            if val not in ("", "0", "1"):
                raise SystemExit(f"item {r['item']} {k}: {val!r} is not 1, 0 or empty")
            r[v] = val
            n += val != ""
        r["note"] = b["note"]
    write(review / SHEET, sheet, list(sheet[0]))
    return n


def selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cols = ["item", "run", "qid", "type", "question", "passages_shown", "answer", "gold_answer",
                "judge_faithful", "judge_correct", "human_faithful (1/0)", "human_correct (1/0)", "note"]
        rows = [dict(zip(cols, [str(i), "R1a", f"q{i}", "factoid", f"Q{i}?", "P", "A", "G", "1", "1", "", "", ""]))
                for i in (1, 2)]
        write(d / SHEET, rows, cols)
        assert make(d) == 2
        header = (d / BLIND).read_text(encoding="utf-8-sig").splitlines()[0]
        assert "judge" not in header, header  # the person never sees the judge's verdict
        b = read(d / BLIND)
        b[0]["human_faithful"], b[0]["human_correct"] = "1", "0"
        write(d / BLIND, b, list(b[0]))
        assert merge(d) == 2
        s = read(d / SHEET)
        assert (s[0]["human_faithful (1/0)"], s[0]["human_correct (1/0)"], s[1]["human_correct (1/0)"]) == ("1", "0", "")
        assert s[0]["judge_faithful"] == "1"  # judge columns untouched
        b[1]["human_correct"] = "yes"
        write(d / BLIND, b, list(b[0]))
        try:
            merge(d)
            raise AssertionError("bad label accepted")
        except SystemExit as e:
            assert "not 1, 0 or empty" in str(e)
        try:
            make(d)
            raise AssertionError("existing blind copy overwritten")
        except SystemExit as e:
            assert "exists" in str(e)
    print("selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "make":
        print(f"wrote {REVIEW / BLIND}: {make(REVIEW)} rows, judge columns left out")
    elif cmd == "merge":
        print(f"merged {merge(REVIEW)} labels into {REVIEW / SHEET}")
    elif cmd == "--selftest":
        selftest()
    else:
        raise SystemExit(__doc__)
