r"""Preregistration §10c, arm A3: a local open-weight LLM (Qwen3-8B Q4_K_M, llama.cpp) judges CRED causality.

Local (labels stay here):
    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_a3_llm.py build {val,fit,test} OUT.jsonl
        unlabelled requests for the reachable rows of a set (fit = the A1 fit rows: train without the 107 positive-only
        abstracts, plus val); one line per pair: pmid, key, gene and disease names, title+abstract
    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_a3_llm.py select SCORES.jsonl
        AUPRC of every prompt on val (condition T), chosen prompt = highest AUPRC (ties -> lower prompt id)
    PYTHONPATH=. .venv/bin/python docs/results/cred/cred_a3_llm.py threshold SCORES.jsonl PROMPT CONDITION
        threshold = max causal F1 on the fit rows (ties -> highest), cred_s10.best_threshold
On the train PC, inside one queue job (the server starts and stops in the same job; one GPU job at a time):
    python cred_a3_llm.py score --server D:\llama\b11026\llama-server.exe --model D:\llama\models\Qwen3-8B-Q4_K_M.gguf
        --requests REQ.jsonl --prompts P1 P2 P3 --conditions T K --out SCORES.jsonl [--limit N]

Score of a pair = p(Yes) / (p(Yes) + p(No)) of the first answer token, greedy, thinking off (empty think block), from
the server's top-20 log-probabilities; if neither token is in the top 20 the score is 0.5 and the row is flagged.
Condition T gives the title and abstract; condition K gives only the two names (memory control). Pure stdlib on the PC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

DEFINITION = ("A gene is causal for a disease if the gene, upon perturbation (including over-expression, knockout, or "
              "any of its mutations), changes the status or severity of the disease.")
EXCLUSIONS = ("The following are NOT causal: a statistical association or correlation between the gene and the disease; "
              "the gene as a risk factor or susceptibility locus; a change in the gene's expression or activity observed "
              "in patients; the gene as a biomarker or as a therapeutic target; the disease activating the gene.")
PROMPTS = {
    "P1": {"system": f"You annotate gene-disease relations in biomedical abstracts. {DEFINITION}",
           "question": "Does the text state that the gene {gene} is causal for the disease {disease}? Answer Yes or No."},
    "P2": {"system": f"You annotate gene-disease relations in biomedical abstracts. {DEFINITION} {EXCLUSIONS}",
           "question": "Does the text state that the gene {gene} is causal for the disease {disease}? Answer Yes or No."},
    "P3": {"system": f"You annotate gene-disease relations in biomedical abstracts. {DEFINITION} {EXCLUSIONS} "
                     "Judge only what the text states, not what you know from elsewhere.",
           "question": "According to this text alone, is the gene {gene} causal for the disease {disease}? "
                       "Answer Yes or No."},
}
K_QUESTION = "Is the gene {gene} causal for the disease {disease}? Answer Yes or No."


def prompt_sha256() -> dict[str, str]:
    return {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k, v in
            {**PROMPTS, "K_QUESTION": K_QUESTION}.items()}


def chat(system: str, user: str) -> str:
    """Qwen3 ChatML with thinking disabled (the empty think block of enable_thinking=False)."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def render(req: dict, prompt: str, condition: str) -> str:
    p = PROMPTS[prompt]
    names = {"gene": req["gene"], "disease": req["disease"]}
    if condition == "K":
        return chat(p["system"], K_QUESTION.format(**names))
    return chat(p["system"], f"Text:\n{req['text']}\n\n" + p["question"].format(**names))


# ---- local: requests, prompt selection, threshold ------------------------------------------------------------------
def names(doc: dict, eid: str) -> str:
    c = Counter(m["text"] for m in doc["mentions"] if eid in m["ids"])
    return " / ".join(t for t, _ in c.most_common(3))


def build(which: str, out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cred_a1_stacker import fit_rows
    from cred_s10 import ROOT, load_rows
    rows = fit_rows() if which == "fit" else [r for r in load_rows(which) if r["status"] == "reachable"]
    docs = {d["pmid"]: d for d in json.loads((ROOT / "data" / "CRED" / "cred_docs.json").read_text(encoding="utf-8"))}
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            assert ";" not in r["pairs"], r
            dis, gene = r["pairs"].split("|")
            d = docs[r["pmid"]]
            f.write(json.dumps({"pmid": r["pmid"], "key": r["pairs"], "gene": names(d, gene), "disease": names(d, dis),
                                "text": d["text"]}, ensure_ascii=False) + "\n")
    print(which, len(rows), "requests ->", out)


def labelled(scores: Path, which: str, prompt: str, condition: str):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cred_a1_stacker import fit_rows
    from cred_s10 import load_rows
    rows = fit_rows() if which == "fit" else [r for r in load_rows(which) if r["status"] == "reachable"]
    got = {}
    for line in open(scores, encoding="utf-8"):
        s = json.loads(line)
        if s["prompt"] == prompt and s["condition"] == condition:
            got[(s["pmid"], s["key"])] = s["score"]
    missing = [r for r in rows if (r["pmid"], r["pairs"]) not in got]
    assert not missing, f"{len(missing)} {which} rows have no {prompt}/{condition} score"
    return np.array([r["label"] == "1" for r in rows]), np.array([got[(r["pmid"], r["pairs"])] for r in rows])


def select(scores: Path) -> None:
    from sklearn.metrics import average_precision_score
    ap = {}
    for p in PROMPTS:
        y, s = labelled(scores, "val", p, "T")
        ap[p] = float(average_precision_score(y, s))
    best = max(PROMPTS, key=lambda p: (ap[p], -int(p[1:])))
    print(json.dumps({"val_auprc_T": ap, "chosen_prompt": best}))


def threshold(scores: Path, prompt: str, condition: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cred_s10 import best_threshold
    y, s = labelled(scores, "fit", prompt, condition)
    t, f1 = best_threshold(y, s)
    print(json.dumps({"prompt": prompt, "condition": condition, "threshold": t, "fit_f1": f1, "n": len(y),
                      "n_causal": int(y.sum())}))


# ---- on the PC: llama-server ----------------------------------------------------------------------------------------
def post(port: int, path: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def yes_no(resp: dict) -> tuple[float, bool]:
    first = resp["completion_probabilities"][0]
    top = first.get("top_logprobs") or first.get("probs") or []
    p = {}
    for t in top:
        tok = (t.get("token") if "token" in t else t.get("tok_str", "")).strip()
        prob = math.exp(t["logprob"]) if "logprob" in t else t["prob"]
        if tok in ("Yes", "No"):
            p[tok] = p.get(tok, 0.0) + prob
    if not p:
        return 0.5, True
    return p.get("Yes", 0.0) / (p.get("Yes", 0.0) + p.get("No", 0.0)), False


def score(a: argparse.Namespace) -> None:
    reqs = [json.loads(line) for line in open(a.requests, encoding="utf-8")][: a.limit]
    server = subprocess.Popen([a.server, "-m", a.model, "-ngl", "99", "-c", "4096", "-np", "1", "--host", "127.0.0.1",
                               "--port", str(a.port)], stdout=subprocess.DEVNULL, stderr=open(a.out + ".server.log", "w"))
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if server.poll() is not None:
                raise RuntimeError(f"llama-server exited with {server.returncode}, see {a.out}.server.log")
            time.sleep(1)
        else:
            raise RuntimeError("llama-server did not become healthy in 600 s")
        for word in ("Yes", "No"):
            toks = post(a.port, "/tokenize", {"content": word})["tokens"]
            assert len(toks) == 1, f"{word!r} is {len(toks)} tokens"
        t0, n_flag = time.time(), 0
        with open(a.out, "w", encoding="utf-8") as f:
            for cond in a.conditions:
                for p in a.prompts:
                    for i, req in enumerate(reqs):
                        resp = post(a.port, "/completion", {"prompt": render(req, p, cond), "n_predict": 1,
                                                            "n_probs": 20, "temperature": 0, "cache_prompt": True})
                        s, flag = yes_no(resp)
                        n_flag += flag
                        f.write(json.dumps({"pmid": req["pmid"], "key": req["key"], "prompt": p, "condition": cond,
                                            "score": s, "not_in_top20": flag}) + "\n")
                        if i % 200 == 0:
                            print(f"{cond} {p} {i}/{len(reqs)} {time.time() - t0:.0f}s", flush=True)
                    if cond == "K":
                        break   # K does not use the text, one prompt system is enough
        meta = {"requests": a.requests, "requests_sha256": hashlib.sha256(Path(a.requests).read_bytes()).hexdigest(),
                "model": a.model, "server": a.server, "prompts_sha256": prompt_sha256(), "n_requests": len(reqs),
                "n_not_in_top20": n_flag, "seconds": round(time.time() - t0)}
        Path(a.out + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(json.dumps(meta))
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("which", choices=("val", "fit", "test"))
    b.add_argument("out")
    s = sub.add_parser("select")
    s.add_argument("scores")
    t = sub.add_parser("threshold")
    t.add_argument("scores")
    t.add_argument("prompt", choices=tuple(PROMPTS))
    t.add_argument("condition", choices=("T", "K"))
    sc = sub.add_parser("score")
    sc.add_argument("--server", required=True)
    sc.add_argument("--model", required=True)
    sc.add_argument("--requests", required=True)
    sc.add_argument("--prompts", nargs="+", choices=tuple(PROMPTS), required=True)
    sc.add_argument("--conditions", nargs="+", choices=("T", "K"), default=["T"])
    sc.add_argument("--out", required=True)
    sc.add_argument("--port", type=int, default=8091)
    sc.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.which, Path(a.out))
    elif a.cmd == "select":
        select(Path(a.scores))
    elif a.cmd == "threshold":
        threshold(Path(a.scores), a.prompt, a.condition)
    else:
        score(a)


if __name__ == "__main__":
    main(sys.argv[1:])
