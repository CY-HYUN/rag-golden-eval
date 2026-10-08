"""Corpus snapshot: arXiv cs.CL / cs.LG papers first submitted after the generator's knowledge cutoff, CC BY only.

Metadata comes from arXiv's OAI-PMH interface (the Atom search API does not return the license); text from the
PDF through pymupdf. The snapshot is written once, hashed, and split into dev / eval slices by paper."""
import gzip
import io
import json
import random
import re
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET

from .common import CORPUS, DATA, sha256

OAI = "https://oaipmh.arxiv.org/oai"
WINDOW = ("2026-07-06", "2026-07-10")  # OAI datestamp window that is listed
CREATED_FROM = "2026-07-01"  # first submitted after the generator's stated knowledge cutoff (2026-06)
PRIMARY = {"cs.CL", "cs.LG"}
SEED, N_SAMPLE, N_KEEP, DEV_SHARE = 20261008, 240, 150, 0.15
MIN_WORDS = 2000
# Each paper is cut to its first 10,000 words (most main texts fit; long appendices do not). This and the 150
# papers keep the dense index to a few hours on this laptop, which embeds about one chunk per second (2026-10-08).
MAX_WORDS = 10_000
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-golden-eval (research snapshot)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def list_candidates() -> list[dict]:
    url = f"{OAI}?verb=ListRecords&metadataPrefix=arXiv&set=cs&from={WINDOW[0]}&until={WINDOW[1]}"
    out, pages = [], 0
    while url:
        root = ET.fromstring(get(url))
        pages += 1
        for m in root.iterfind(".//a:arXiv", NS):
            f = {k: (m.findtext(f"a:{k}", "", NS) or "").strip() for k in ("id", "created", "categories", "license", "title", "abstract")}
            cats = f["categories"].split()
            # OAI "created" can be a later version's date (2602.10099 shows 2026-07-03), so the id's yymm,
            # which arXiv assigns at first submission, has to agree too
            if (cats and cats[0] in PRIMARY and f["created"] >= CREATED_FROM and f["id"][:4] >= CREATED_FROM[2:4] + CREATED_FROM[5:7]
                    and f["license"].startswith("http://creativecommons.org/licenses/by/")):
                f["authors"] = [" ".join(filter(None, (a.findtext("a:forenames", "", NS), a.findtext("a:keyname", "", NS))))
                                for a in m.iterfind("a:authors/a:author", NS)]
                f["title"], f["abstract"] = (" ".join(f[k].split()) for k in ("title", "abstract"))
                out.append(f)
        tok = root.findtext(".//o:resumptionToken", "", NS)
        url = f"{OAI}?verb=ListRecords&resumptionToken={tok}" if tok else None
        print(f"page {pages}: {len(out)} CC BY candidates so far", flush=True)
        time.sleep(3)
    return sorted({p["id"]: p for p in out}.values(), key=lambda p: p["id"])


def clean(raw: str) -> str:
    """NFKC (ligatures), join hyphenated line breaks, drop the reference list when it sits in the second half,
    collapse whitespace. Character offsets everywhere refer to this cleaned text."""
    t = unicodedata.normalize("NFKC", raw).replace("\x00", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    refs = [m.start() for m in re.finditer(r"(?m)^\s*(References|REFERENCES|Bibliography)\s*$", t)]
    if refs and refs[-1] > 0.5 * len(t):
        t = t[:refs[-1]]
    return " ".join(t.split())


def pdf_text(pid: str) -> str:
    import pymupdf
    path = DATA / "pdf" / f"{pid}.pdf"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get(f"https://arxiv.org/pdf/{pid}"))
        time.sleep(3)
    with pymupdf.open(stream=io.BytesIO(path.read_bytes()), filetype="pdf") as doc:
        return clean("\n".join(page.get_text("text") for page in doc))


def build() -> None:
    import pymupdf
    cands = list_candidates()
    sample = random.Random(SEED).sample(cands, min(N_SAMPLE, len(cands)))
    kept, dropped = [], []
    for p in sample:
        if len(kept) == N_KEEP:
            break
        try:
            text = pdf_text(p["id"])
        except Exception as e:  # recorded per paper in the manifest, never silently skipped
            dropped.append({"id": p["id"], "why": f"{type(e).__name__}: {e}"[:200]})
            continue
        if len(text.split()) < MIN_WORDS:
            dropped.append({"id": p["id"], "why": f"{len(text.split())} words < {MIN_WORDS}"})
            continue
        kept.append({**p, "text": " ".join(text.split()[:MAX_WORDS])})
        print(len(kept), p["id"], len(text.split()), "words", flush=True)
    kept.sort(key=lambda p: p["id"])
    body = "".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in kept).encode("utf-8")
    raw = DATA / "corpus.jsonl"
    raw.write_bytes(body)
    with open(CORPUS, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g:
        g.write(body)
    ids = [p["id"] for p in kept]
    dev = sorted(random.Random(SEED + 1).sample(ids, round(DEV_SHARE * len(ids))))
    (DATA / "split.json").write_text(json.dumps({"seed": SEED + 1, "dev": dev, "eval": sorted(set(ids) - set(dev))}, indent=1), newline="\n")
    (DATA / "manifest.json").write_text(json.dumps({
        "source": f"{OAI} ListRecords metadataPrefix=arXiv set=cs from={WINDOW[0]} until={WINDOW[1]}",
        "filter": f"primary category in {sorted(PRIMARY)}, created >= {CREATED_FROM}, license http://creativecommons.org/licenses/by/*",
        "built": time.strftime("%Y-%m-%d"), "pymupdf": pymupdf.VersionBind, "seed": SEED,
        "max_words_per_paper": MAX_WORDS, "candidates": len(cands), "sampled_tried": len(kept) + len(dropped), "kept": len(kept), "dropped": dropped,
        "corpus_jsonl_sha256": sha256(raw), "words": sum(len(p["text"].split()) for p in kept)}, indent=1), newline="\n")
    raw.unlink()  # the committed artifact is the gzip; its uncompressed content is what the hash covers
    print(f"kept {len(kept)}, dropped {len(dropped)}, candidates {len(cands)}")
