"""uv run python -m rge <command>

    corpus      fetch, filter, extract and hash the snapshot; write the dev / eval split
    golden      draft questions from the eval slice, check each against its own evidence, write golden_set_v1
    index       build the BM25 and dense (Qdrant local) indexes over every chunk
    measure     run the ladder (R0-R4, R1 three times) through generation and the judge
    repeat      generate R3 and R4 once more (new batch) and judge them; rows R3-rep, R4-rep next to the originals
    judge-sensitivity [model ...]  re-judge R1a-R4 with other judge models; results/judge_sensitivity.md
    report      print the results table from results/ (no API calls)
    calibrate   agreement and Cohen's kappa of the judge against review/judge_calibration.csv
    selftest    metric functions on hand-computed examples
"""
import sys


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "corpus":
        from .corpus import build
        build()
    elif cmd == "golden":
        from .golden import build
        build()
    elif cmd == "index":
        from .retrieve import build_indexes
        build_indexes()
    elif cmd == "measure":
        from .harness import measure
        measure()
    elif cmd == "repeat":
        from .harness import measure
        measure(["R3", "R4"], "generate-v1-rep", "-rep")
    elif cmd == "judge-sensitivity":
        from .harness import JUDGE_ALT, judge_sensitivity
        judge_sensitivity(sys.argv[2:] or JUDGE_ALT)
    elif cmd == "report":
        from .harness import report
        report()
    elif cmd == "calibrate":
        from .harness import calibrate
        calibrate()
    elif cmd == "selftest":
        from .selftest import run
        run()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
