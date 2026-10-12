"""Stage 2.7: latency + token cost per question, per system, from traced eval runs.

Run: .venv/bin/python stage2/ops.py --label ops        (uses stage2/out/runs/<label>/run_*.json)
Cost: tokens only. Gateway prices are unknown here; multiply by your provider's $/token.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles

RUNS = Path(__file__).parent / "out" / "runs"


def pct(xs, p):
    return quantiles(xs, n=100)[p - 1] if len(xs) > 1 else xs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="ops")
    results = [r for p in sorted((RUNS / ap.parse_args().label).glob("run_*.json")) for r in json.loads(p.read_text())]
    by_sys = defaultdict(list)
    for r in results:
        by_sys[r["system"]].append(r)

    print(f"{'system':9} {'n':>4} {'p50 s':>6} {'p95 s':>6} {'LLM calls':>9} {'tok in':>7} {'tok out':>7} {'LLM s':>6} {'DB s':>6} {'embed s':>7}")
    for s, rs in by_sys.items():
        llm = [[t for t in r["trace"] if t["kind"] == "llm"] for r in rs]
        secs = [r["seconds"] for r in rs]
        part = lambda kind: mean(sum(t["seconds"] for t in r["trace"] if t["kind"] == kind) for r in rs)
        print(f"{s:9} {len(rs):>4} {pct(secs, 50):>6.1f} {pct(secs, 95):>6.1f} {mean(map(len, llm)):>9.1f} "
              f"{mean(sum(t['tokens_in'] for t in c) for c in llm):>7.0f} {mean(sum(t['tokens_out'] for t in c) for c in llm):>7.0f} "
              f"{part('llm'):>6.2f} {part('neo4j'):>6.3f} {part('embed'):>7.3f}")

    print("\nper LLM purpose (all systems):")
    purposes = defaultdict(list)
    for r in results:
        for t in r["trace"]:
            if t["kind"] == "llm":
                purposes[t["purpose"]].append(t)
    for p, ts in sorted(purposes.items(), key=lambda x: -sum(t["tokens_in"] for t in x[1])):
        print(f"  {p:8} calls {len(ts):4}  mean {mean(t['seconds'] for t in ts):4.1f}s  "
              f"mean tokens in {mean(t['tokens_in'] for t in ts):6.0f} out {mean(t['tokens_out'] for t in ts):4.0f}")


if __name__ == "__main__":
    main()
