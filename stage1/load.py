"""Stage 1.4: load entities + triples into Neo4j (full reload, idempotent).

Run: .venv/bin/python stage1/load.py [out_dir]   (default stage1/out; needs: docker compose up -d)
Browse: http://localhost:7474  (user neo4j, password NEO4J_PASSWORD from .env)
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

import env  # noqa: F401  (loads .env)
from schema import NODES, validate

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out"


def merge_triples(lines):
    """Same fact from several docs -> one edge whose `sources` lists every doc."""
    edges = {}
    for t in map(json.loads, lines):
        key = (t["subject_type"], t["relation"], t["object_type"], t["subject"], t["object"])
        e = edges.setdefault(key, {"subject": t["subject"], "object": t["object"],
                                   "role": None, "since": None, "until": None, "sources": []})
        e["sources"].append(t["source"])
        e["role"] = e["role"] or t["role"]
        e["since"] = e["since"] or t["since"]
        e["until"] = e["until"] or t.get("until")
    return edges


def main():
    entities = json.loads((OUT / "entities.json").read_text())
    edges = merge_triples(open(OUT / "triples.jsonl"))
    by_shape = defaultdict(list)
    for (st, rel, ot, _, _), e in edges.items():
        by_shape[(st, rel, ot)].append(e)

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.environ["NEO4J_PASSWORD"]))
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")  # full reload: graph = exactly what extraction produced
        for label in NODES:
            s.run(f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE")

        for e in entities:
            assert e["type"] in NODES  # labels are pasted into Cypher text -> whitelist first
            s.run(f"MERGE (n:{e['type']} {{name: $name}}) SET n.aliases = $aliases, n.title = $title",
                  name=e["name"], aliases=e["aliases"], title=e.get("title"))

        for (st, rel, ot), rows in by_shape.items():
            assert validate(st, rel, ot) is None  # relation/labels pasted into Cypher -> whitelist first
            s.run(f"""UNWIND $rows AS r
                      MERGE (a:{st} {{name: r.subject}})
                      MERGE (b:{ot} {{name: r.object}})
                      MERGE (a)-[e:{rel}]->(b)
                      SET e.sources = r.sources, e.role = r.role, e.since = r.since, e.until = r.until""",
                  rows=rows)

        nodes = s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label").data()
        rels = s.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel").data()
        print("nodes:", {r["label"]: r["n"] for r in nodes}, "total", sum(r["n"] for r in nodes))
        print("edges:", {r["rel"]: r["n"] for r in rels}, "total", sum(r["n"] for r in rels))

        # Smoke test: the Stage 0.1 question, now as one Cypher query.
        q = """MATCH (p:Person)-[:LEADS]->(t:Team)-[o:OWNS]->(:Project {name: 'Atlas'})
               RETURN p.name AS leader, t.name AS team, o.sources AS sources"""
        print("\nWho leads the team that owns Atlas?", s.run(q).data())
    driver.close()


if __name__ == "__main__":
    main()
