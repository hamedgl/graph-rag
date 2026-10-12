"""Stage 2.6: evaluation at scale = repeated runs + faithfulness + regression gate.

  run   --label L -n N   N parallel eval runs -> stage2/out/runs/L/run_K.json
  report --label L       mean/min/max per system, flaky questions (pass in some runs, fail in others)
  faith  --label L       judge (JUDGE_MODEL, a different model than the answerer) checks every answer's
                         claims against its retrieved context -> healthy / lucky / retrieval-problem / hallucination
  gate  --base B --new L exit 1 if the router mean drops by > 1 question, or a question that passed in
                         EVERY base run now fails in EVERY new run

Run: .venv/bin/python stage2/eval_scale.py run --label baseline -n 3
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "stage1"))
from extract import obj  # noqa: E402  (loads .env)
from llm import call_json  # noqa: E402

RUNS = ROOT / "stage2" / "out" / "runs"
QUESTIONS = ROOT / "stage2" / "data" / "questions_v3.json"
DOCS = [ROOT / "stage1" / "data" / "docs", ROOT / "stage2" / "data" / "docs_extra", ROOT / "stage2" / "data" / "docs_new"]
SYSTEMS = ["vector", "traverse", "cypher", "router"]
JUDGE_MODEL = os.environ["JUDGE_MODEL"]


def load(label):
    return [json.loads(p.read_text()) for p in sorted((RUNS / label).glob("run_*.json"))]


def run(label, n):
    (RUNS / label).mkdir(parents=True, exist_ok=True)
    for k in range(1, n + 1):
        # ponytail: runs are sequential, each internally parallel (8 workers); parallel runs would
        # multiply gateway load
        r = subprocess.run([sys.executable, str(ROOT / "stage1" / "eval.py"), "--questions", str(QUESTIONS),
                        "--docs", *map(str, DOCS), "--systems", *SYSTEMS, "--workers", "8",
                        "--out", str(RUNS / label / f"run_{k}.json")], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"run {k} failed:\n{r.stderr[-2000:]}")  # surface the real error, don't swallow it
        errors = [line for line in r.stdout.splitlines() if line.startswith("ERRORS")]
        print(f"  run {k}/{n} done {errors[0] if errors else '(0 errors)'}", flush=True)


def scores(runs):
    """system -> [correct count per run]; (question, system) -> [pass per run]."""
    per_run, per_q = defaultdict(list), defaultdict(list)
    for results in runs:
        tally = defaultdict(int)
        for r in results:
            tally[r["system"]] += r["correct"]
            per_q[(r["q"], r["system"])].append(r["correct"])
        for s in SYSTEMS:
            per_run[s].append(tally[s])
    return per_run, per_q


def report(label):
    runs = load(label)
    per_run, per_q = scores(runs)
    n_q = len({r["q"] for r in runs[0]})
    print(f"{label}: {len(runs)} runs x {n_q} questions")
    for s in SYSTEMS:
        xs = per_run[s]
        print(f"  {s:9} mean {mean(xs):5.1f}/{n_q}  min {min(xs)}  max {max(xs)}  runs {xs}")
    flaky = [(q, s, xs) for (q, s), xs in per_q.items() if 0 < sum(xs) < len(xs) and s in ("cypher", "router")]
    print(f"  flaky (cypher/router, pass in some runs only): {len(flaky)}")
    for q, s, xs in flaky:
        print(f"    [{s}] {q}  {['✓' if x else '✗' for x in xs]}")


JUDGE = """You check an answer against the context it was generated from.
List every factual claim in the ANSWER that the CONTEXT does not support (or contradicts).
Statements that information is missing/unknown are supported if the context indeed lacks it.
supported=true only if there are no unsupported claims."""
JUDGE_SCHEMA = obj(supported={"type": "boolean"}, unsupported_claims={"type": "array", "items": {"type": "string"}})


def judge(r):
    out = call_json(JUDGE_MODEL, JUDGE, f"CONTEXT:\n{r.get('context') or '(empty)'}\n\nQUESTION: {r['q']}\n\n"
                                        f"ANSWER: {r['answer']}", "judge", JUDGE_SCHEMA)
    return out.get("supported", False), out.get("unsupported_claims", [])


def faith(label):
    path = sorted((RUNS / label).glob("run_*.json"))[-1]  # latest run of the label
    results = json.loads(path.read_text())
    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(judge, results))
    groups = defaultdict(lambda: defaultdict(list))
    for r, (ok, claims) in zip(results, verdicts):
        r["faithful"], r["unsupported_claims"] = ok, claims
        g = {(True, True): "healthy", (True, False): "lucky", (False, True): "retrieval/data problem",
             (False, False): "hallucination"}[(r["correct"], ok)]
        groups[r["system"]][g].append(r)
    path.write_text(json.dumps(results, indent=2))
    print(f"faithfulness ({path.name}, judge {JUDGE_MODEL}):")
    for s in SYSTEMS:
        print(f"  {s:9} " + "  ".join(f"{g}: {len(groups[s][g])}" for g in
                                      ("healthy", "lucky", "retrieval/data problem", "hallucination")))
    for s in SYSTEMS:
        for g in ("hallucination", "lucky"):
            for r in groups[s][g]:
                print(f"    [{s}] {g.upper()}: {r['q']}\n        answer: {r['answer'][:150]}\n"
                      f"        unsupported: {r['unsupported_claims']}")


def gate(base, new):
    (b_run, b_q), (n_run, n_q) = scores(load(base)), scores(load(new))
    drop = mean(b_run["router"]) - mean(n_run["router"])
    broken = [q for (q, s), xs in b_q.items() if s == "router" and all(xs) and not any(n_q.get((q, s), [True]))]
    print(f"gate {base} -> {new}: router mean {mean(b_run['router']):.1f} -> {mean(n_run['router']):.1f}")
    for q in broken:
        print(f"  REGRESSION (always passed, now always fails): {q}")
    failed = drop > 1 or broken
    print("GATE: FAIL" if failed else "GATE: PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "faith", "gate"])
    ap.add_argument("--label", default="baseline")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--base")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    a = ap.parse_args()
    QUESTIONS = a.questions
    {"run": lambda: run(a.label, a.n), "report": lambda: report(a.label), "faith": lambda: faith(a.label),
     "gate": lambda: gate(a.base, a.label)}[a.cmd]()
