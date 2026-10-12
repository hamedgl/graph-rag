"""Print the graph facts about given people (incl. sources/since/until). Run: show.py NAME..."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
from retrieve import db, read  # noqa: E402

for name in sys.argv[1:]:
    rows = read("""MATCH (p {name: $n})-[r]-(m) WHERE NOT m:Document
                   RETURN startNode(r).name AS a, type(r) AS rel, endNode(r).name AS b,
                          r.since AS since, r.until AS until, r.sources AS src ORDER BY rel""", n=name)
    print(f"{name}: {len(rows)} edges" if rows else f"{name}: NOT IN GRAPH")
    for r in rows:
        t = "".join(f" {k}={r[k]}" for k in ("since", "until") if r[k])
        print(f"   {r['a']} -[{r['rel']}{t}]-> {r['b']}   {r['src']}")
db.close()
