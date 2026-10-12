"""Stage 1.3: score extracted triples against ground truth (free, no API calls).

Run: python3 stage1/score.py
Exact match on (subject, relation, object). A name variant ("A. Chen") counts as wrong:
that is the point, the graph would treat it as a different node.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def score(truth, pred):
    tp = truth & pred
    precision = len(tp) / len(pred) if pred else 0.0
    recall = len(tp) / len(truth) if truth else 0.0
    return precision, recall, pred - truth, truth - pred


def main():
    truth = {(f["subject"], f["relation"], f["object"]) for f in csv.DictReader(open(HERE / "data" / "facts.csv"))}
    sources = defaultdict(set)
    for line in open(HERE / "out" / "triples.jsonl"):
        t = json.loads(line)
        sources[(t["subject"], t["relation"], t["object"])].add(t["source"])
    pred = set(sources)

    precision, recall, wrong, missed = score(truth, pred)
    print(f"truth {len(truth)} | extracted {len(pred)} unique | precision {precision:.0%} | recall {recall:.0%}")
    print(f"\nWRONG (in graph, not true) - {len(wrong)}:")
    for s, r, o in sorted(wrong, key=lambda x: x[1]):
        print(f"  {s} -[{r}]-> {o}   from {', '.join(sorted(sources[(s, r, o)]))}")
    print(f"\nMISSED (true, not in graph) - {len(missed)}:")
    for s, r, o in sorted(missed, key=lambda x: x[1]):
        print(f"  {s} -[{r}]-> {o}")


if __name__ == "__main__":
    t = {("a", "R", "b"), ("c", "R", "d")}
    assert score(t, {("a", "R", "b"), ("x", "R", "y")})[:2] == (0.5, 0.5)
    main()
