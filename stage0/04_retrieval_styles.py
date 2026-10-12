"""Stage 0.4 demo: 4 retrieval styles, each on the question type it is built for.

Run: python3 stage0/04_retrieval_styles.py  (stdlib only)
LLM steps are shown as the text they would produce; Stage 1 makes them real.
"""
from collections import Counter, defaultdict

EDGES = [
    ("Alice Chen", "LEADS", "Payments"), ("Dave Lee", "LEADS", "Search"),
    ("Bob Kumar", "MEMBER_OF", "Payments"), ("Carol Diaz", "MEMBER_OF", "Payments"),
    ("Eve Park", "MEMBER_OF", "Search"),
    ("Payments", "OWNS", "Atlas"), ("Payments", "OWNS", "Ledger"), ("Search", "OWNS", "Orion"),
    ("Atlas", "USES", "Kafka"), ("Ledger", "USES", "Postgres"), ("Orion", "USES", "Kafka"),
]
NODES = {n for s, _, o in EDGES for n in (s, o)}


def neighbors(seeds, hops=2):
    frontier, found = set(seeds), set()
    for _ in range(hops):
        step = {e for e in EDGES if e[0] in frontier or e[2] in frontier}
        found |= step
        frontier = {n for s, _, o in step for n in (s, o)}
    return sorted(found)


# 1. ENTITY TRAVERSAL: question names an entity -> start there, walk out
q = "Who works on Atlas?"
seeds = {n for n in NODES if n in q}
print(f"[1 Traversal] {q}\n  seeds={seeds}")
for e in neighbors(seeds):
    print("  ", e)

# 2. TEXT2CYPHER: aggregation/filter -> LLM writes a query, DB computes exact answer
q = "How many members does each team have?"
cypher = "MATCH (p:Person)-[:MEMBER_OF]->(t:Team) RETURN t.name, count(p)"
counts = Counter(o for s, r, o in EDGES if r == "MEMBER_OF")  # what the DB would compute
print(f"\n[2 Text2Cypher] {q}\n  LLM writes: {cypher}\n  DB returns: {dict(counts)}")

# 3. HYBRID: question uses no exact node name -> fuzzy/vector match finds the seed
q = "Who manages the payment team?"
exact = {n for n in NODES if n in q}
# ponytail: shared-prefix match stands in for vector similarity on node names
fuzzy = {n for n in NODES if any(w[:5].lower() == n[:5].lower() for w in q.split())}
print(f"\n[3 Hybrid] {q}\n  exact match seeds={exact or 'NONE'}  -> vector match seeds={fuzzy}")
print("  ", [e for e in neighbors(fuzzy, hops=1) if e[1] == "LEADS"])

# 4. COMMUNITY SUMMARIES: global question, no seed entity -> pre-summarized clusters
q = "What are the main technical themes across the company?"
print(f"\n[4 Community] {q}\n  seeds={ {n for n in NODES if n in q} or 'NONE (traversal has nowhere to start)'}")
communities = defaultdict(set)  # ponytail: cluster = team + what it owns/uses; real systems run Leiden
for team in ("Payments", "Search"):
    for s, _, o in neighbors({team}):
        communities[team] |= {s, o}
for team, members in communities.items():
    techs = sorted(m for m in members if m in ("Kafka", "Postgres"))
    print(f"  summary[{team}] (written by LLM at index time): projects + tech {techs}")
print("  -> answer LLM reads the summaries, not raw edges")
