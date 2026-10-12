"""Stage 2.5: merge a structured HR export into the document-built graph (no LLM).

Policy (user choice): HR wins for its fields, UNLESS a document claim is explicitly newer
(edge.since > export year). Every disagreement is logged; the losing edge is kept and flagged
overridden_by=<hr file>. Nothing is resolved silently.

Identity: employee_id first, then exact name/alias, else a new Person (HR IDs = ER anchors).
Run: .venv/bin/python stage2/hr_import.py stage2/data/hr/hr_export_2025-12-01.csv
     -> stage2/out/hr_conflicts.json
"""
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
import env  # noqa: E402,F401  (loads .env)

db = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.environ["NEO4J_PASSWORD"]))


def match_person(s, row):
    """employee_id -> exact name/alias -> new. Returns (canonical name, how)."""
    r = s.run("MATCH (p:Person {employee_id: $id}) RETURN p.name AS n", id=row["employee_id"]).single()
    if r:
        return r["n"], "employee_id"
    r = s.run("""MATCH (p:Person) WHERE toLower(p.name) = toLower($n)
                    OR any(a IN coalesce(p.aliases, []) WHERE toLower(a) = toLower($n))
                 RETURN p.name AS n""", n=row["name"]).single()
    if r:
        return r["n"], "name"
    s.run("MERGE (p:Person {name: $n})", n=row["name"])
    return row["name"], "new (HR only)"


def add_source(s, rel, label, a, b, src):
    s.run(f"""MATCH (a:Person {{name: $a}}) MERGE (b:{label} {{name: $b}})
              MERGE (a)-[e:{rel}]->(b)
              SET e.sources = CASE WHEN $src IN coalesce(e.sources, []) THEN e.sources
                              ELSE coalesce(e.sources, []) + $src END""", a=a, b=b, src=src)


def apply_field(s, rel, label, person, target, src, as_of_year, conflicts):
    """HR asserts `person -[rel]-> target` as current. Compare with the graph's current edges."""
    # rel and label come from the fixed list in main(), never from the CSV: safe to place in query text
    current = s.run(f"""MATCH (:Person {{name: $p}})-[e:{rel}]->(b:{label})
                        WHERE e.until IS NULL AND e.overridden_by IS NULL
                        RETURN b.name AS b, e.since AS since""", p=person).data()
    if any(c["b"] == target for c in current):
        add_source(s, rel, label, person, target, src)
        return
    if not current:  # HR fills a gap: nothing to disagree with
        add_source(s, rel, label, person, target, src)
        return
    newer = [c for c in current if c["since"] and c["since"] > as_of_year]
    if newer:  # documents are explicitly newer than the export: they win, HR value becomes history
        add_source(s, rel, label, person, target, src)
        s.run(f"""MATCH (:Person {{name: $p}})-[e:{rel}]->(:{label} {{name: $b}})
                  SET e.until = coalesce(e.until, $until)""", p=person, b=target, until=min(c["since"] for c in newer))
        winner = "documents (newer)"
    else:  # undated or older doc claims: HR is authoritative for its own fields
        add_source(s, rel, label, person, target, src)
        s.run(f"""MATCH (:Person {{name: $p}})-[e:{rel}]->(b:{label})
                  WHERE e.until IS NULL AND b.name <> $t SET e.overridden_by = $src""", p=person, t=target, src=src)
        winner = "HR"
    conflicts.append({"person": person, "field": rel, "hr": target, "docs": [c["b"] for c in current],
                      "winner": winner})


def main(path):
    path = Path(path)
    src = path.name
    as_of = re.search(r"\d{4}-\d{2}-\d{2}", src).group()
    as_of_year = int(as_of[:4])
    rows = list(csv.DictReader(open(path)))
    conflicts, how = [], {}
    with db.session() as s:
        canon = {}
        for row in rows:
            canon[row["employee_id"]], how[row["name"]] = match_person(s, row)
            s.run("MATCH (p:Person {name: $n}) SET p.employee_id = $id, p.email = $e",
                  n=canon[row["employee_id"]], id=row["employee_id"], e=row["email"])
        for row in rows:
            p = canon[row["employee_id"]]
            if row["title"]:
                old = s.run("MATCH (p:Person {name: $n}) RETURN p.title AS t", n=p).single()["t"]
                if old and old.lower() != row["title"].lower():
                    conflicts.append({"person": p, "field": "title", "hr": row["title"], "docs": [old], "winner": "HR"})
                s.run("MATCH (p:Person {name: $n}) SET p.title = $t", n=p, t=row["title"])
            for rel, label, target in (("LEADS" if row["is_lead"] == "yes" else "MEMBER_OF", "Team", row["team"]),
                                       ("REPORTS_TO", "Person", canon.get(row["manager_id"])),
                                       ("LOCATED_IN", "Office", row["office"])):
                if target:
                    apply_field(s, rel, label, p, target, src, as_of_year, conflicts)
        s.run("""MERGE (d:Document {name: $n})
                 SET d.kind = 'structured', d.as_of = $a, d.ingested_at = $now""",
              n=src, a=as_of, now=datetime.now(timezone.utc).isoformat())

    print(f"{src} (as of {as_of}): {len(rows)} rows")
    print("  identity:", {k: sum(1 for v in how.values() if v == k) for k in set(how.values())})
    print(f"  HR-only people: {[n for n, h in how.items() if h.startswith('new')]}")
    print(f"  conflicts: {len(conflicts)}")
    for c in conflicts:
        print(f"    {c['person']:12} {c['field']:10} HR={c['hr']!r:22} docs={c['docs']}  -> {c['winner']}")
    out = Path(__file__).parent / "out" / "hr_conflicts.json"
    out.write_text(json.dumps(conflicts, indent=2))
    db.close()


if __name__ == "__main__":
    main(sys.argv[1])
