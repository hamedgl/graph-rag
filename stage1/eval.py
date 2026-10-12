"""Stage 1.7: Graph RAG (Text2Cypher, traversal) vs plain vector RAG on the same questions.

Run: .venv/bin/python stage1/eval.py [--questions Q.json] [--docs DIR ...] [--out OUT.json]
Same docs, same answer LLM, same instructions; only retrieval differs.
Scoring is code, not an LLM: expected names must appear, forbidden ones must not,
'not-found' questions must come back known=false. Details -> stage1/out/eval.json.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import time

import retrieve
from llm import get_trace, start_trace
from answer import answer
from retrieve import db

HERE = Path(__file__).parent
SYSTEMS = ["vector", "traverse", "cypher"]


def has(text, term):
    return re.search(rf"\b{re.escape(term.lower())}\b", text.lower()) is not None


def correct(q, a):
    if a.get("error"):  # an error is never a correct answer (not even a correct 'not found')
        return False
    if not q["expect"]:  # not-found question: must admit it
        return not a["known"]
    return (a["known"]
            and all(any(has(a["answer"], alt) for alt in e.split("|")) for e in q["expect"])
            and not any(has(a["answer"], f) for f in q["forbid"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=HERE / "data" / "questions.json", type=Path)
    ap.add_argument("--docs", nargs="+", type=Path)
    ap.add_argument("--out", default=HERE / "out" / "eval.json", type=Path)
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=[*SYSTEMS, "router"])
    ap.add_argument("--workers", type=int, default=8, help="parallel (question, system) jobs; mind gateway rate limits")
    args = ap.parse_args()
    if args.docs:
        retrieve.DOC_DIRS = args.docs
    questions = json.loads(args.questions.read_text())
    from router import routed_answer  # import before threads start
    if {"vector", "router"} & set(args.systems):
        retrieve.vector_search("warm-up")  # load the embedding model once, not inside racing threads

    def run(job):
        i, system = job
        q = questions[i]
        start_trace()  # one job = one thread: everything it calls is recorded in this trace
        t = time.perf_counter()
        try:
            a = routed_answer(q["q"]) if system == "router" else answer(q["q"], system)
        except Exception as e:  # gateway timeout, rate limit...: an ops error, not a crash of the whole run
            a = {"answer": "", "citations": [], "known": False, "retrieval": "error",
                 "error": f"{type(e).__name__}: {e}"[:300]}
        a["trace"], a["seconds"] = get_trace(), round(time.perf_counter() - t, 2)
        return job, a

    jobs = [(i, s) for i in range(len(questions)) for s in args.systems]
    with ThreadPoolExecutor(args.workers) as pool:
        done = dict(pool.map(run, jobs))  # (question, system) -> answer; jobs run in any order

    results, table = [], defaultdict(lambda: defaultdict(int))
    for i, q in enumerate(questions):
        marks = []
        for system in args.systems:
            a = done[(i, system)]
            ok = correct(q, a)
            table[q["type"]][system] += ok
            marks.append(f"{system} {'✓' if ok else '✗'}")
            results.append({**q, "system": system, "correct": ok, **a})
        print(f"{i + 1:2}. [{q['type']:11}] {q['q']:60} {' | '.join(marks)}")

    args.out.write_text(json.dumps(results, indent=2))
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\nERRORS: {len(errors)} jobs failed")
        for r in errors:
            print(f"  [{r['system']}] {r['q']}: {r['error']}")
    counts = defaultdict(int)
    for q in questions:
        counts[q["type"]] += 1
    print(f"\n{'type':12}{'n':>3}  " + "".join(f"{s:>10}" for s in args.systems))
    for t, n in counts.items():
        print(f"{t:12}{n:>3}  " + "".join(f"{table[t][s]:>7}/{n:<2}" for s in args.systems))
    total = len(questions)
    print(f"{'TOTAL':12}{total:>3}  " + "".join(f"{sum(table[t][s] for t in counts):>7}/{total:<2}" for s in args.systems))
    db.close()


if __name__ == "__main__":
    a = {"answer": "Bob Kumar and Carol Diaz, since 2024.", "known": True}
    assert correct({"expect": ["Bob Kumar", "Carol Diaz"], "forbid": ["Grace Liu"]}, a)
    assert not correct({"expect": ["4|four"], "forbid": []}, a)  # '2024' must not count as '4'
    assert correct({"expect": [], "forbid": []}, {"answer": "Not found.", "known": False})
    main()
