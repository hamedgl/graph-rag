"""Stage 2.3: incremental ingestion (add / update / delete docs without a full rebuild).

Registry: (:Document {name, hash, version, ingested_at}) nodes in the same Neo4j (same transaction as edges).
Provenance: every edge has e.sources = [doc names]. Delete a doc -> drop it from sources; delete edges
left with no sources; delete nodes left with no edges.
Incremental ER: each new surface name -> exact name/alias match in the graph, else closed-world LLM
linking against graph candidates (stage2/resolve.py link), else a new entity.

Run: .venv/bin/python stage2/ingest.py bootstrap DIR...   register docs already in the graph (no extraction)
     .venv/bin/python stage2/ingest.py sync DIR...        add new/changed docs, remove deleted ones
     .venv/bin/python stage2/ingest.py remove DOC_NAME
     .venv/bin/python stage2/ingest.py status
"""
import difflib
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent))
import resolve as R  # noqa: E402  (also puts stage1 on sys.path, loads .env)
from llm import INGEST_MODEL, call_json  # noqa: E402
from schema import NODES, SINGLE_CURRENT, validate  # noqa: E402

db = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.environ["NEO4J_PASSWORD"]))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def files(dirs):
    return {p.name: p for d in dirs for p in sorted(Path(d).glob("*.md"))}


# ---------- incremental entity resolution ----------
def graph_entities(s, label):
    return s.run(f"MATCH (n:{label}) RETURN n.name AS name, coalesce(n.aliases, []) AS aliases").data()


def graph_context(s, name):
    rows = s.run("""MATCH (n {name: $name})-[r]-(m) WHERE NOT m:Document
                    RETURN startNode(r).name AS a, type(r) AS rel, endNode(r).name AS b LIMIT 12""", name=name).data()
    return f"{name}. Facts: " + "; ".join(f"{r['a']} {r['rel']} {r['b']}" for r in rows)


def resolve_name(s, surface, label, doc_facts, doc_text):
    """Return (canonical name, how). label is already whitelisted by validate()."""
    ents = graph_entities(s, label)
    for e in ents:  # 1. exact name or known alias: free
        if surface.lower() in [e["name"].lower(), *map(str.lower, e["aliases"])]:
            return e["name"], "exact"
    me = {"type": label, "name": surface}
    cands = []
    for e in ents:  # 2. blocking against graph entities of the same label
        # ponytail: scans every entity of the label; at scale use a fulltext index on name/aliases
        other = {"type": label, "name": e["name"]}
        na, nb = R.norm(surface), R.norm(e["name"])
        if R.is_full_name(me) and R.is_full_name(other) and difflib.SequenceMatcher(None, na, nb).ratio() < 0.85:
            continue  # two different full names: never the same (same rule as resolve.py)
        if set(na.split()) & set(nb.split()) or difflib.SequenceMatcher(None, na, nb).ratio() > 0.75:
            cands.append(e["name"])
    if cands:  # 3. closed-world LLM linking
        sents = [x for x in doc_text.splitlines() if surface.lower() in x.lower()]
        mention = {"name": surface, "context": f"{surface} ({label}). Facts: {'; '.join(doc_facts)}. "
                                                f"Text: {' | '.join(sents)}"}
        k, why = R.link(mention, [{"context": graph_context(s, c)} for c in cands])
        if k is not None:
            return cands[k], f"linked from {cands}"
    return surface, "new entity"


# ---------- add / remove ----------
def remove(s, doc):
    n = s.run("""MATCH ()-[e]->() WHERE $doc IN e.sources
                 SET e.sources = [x IN e.sources WHERE x <> $doc]
                 WITH e WHERE size(e.sources) = 0 DELETE e RETURN count(*) AS n""", doc=doc).single()["n"]
    orphans = s.run("MATCH (n) WHERE NOT n:Document AND NOT (n)--() DETACH DELETE n RETURN count(*) AS n").single()["n"]
    s.run("MATCH (d:Document {name: $doc}) DELETE d", doc=doc)
    # ponytail: property-level provenance is NOT tracked: an `until` set by this doc stays after removal.
    print(f"  removed {doc}: {n} edges lost their last source, {orphans} orphan nodes deleted")


def add(s, path):
    text = path.read_text()
    out = call_json(INGEST_MODEL, R.MENTION_INSTRUCTIONS, text, "triples", R.TRIPLES_SCHEMA)
    triples = []
    for t in out["triples"]:
        props = [k for k in ("role", "since", "until") if t.get(k) is not None]
        if validate(t["subject_type"], t["relation"], t["object_type"], props) is None:
            triples.append(t)
    facts = [f"{t['subject']} {t['relation']} {t['object']}" for t in triples]
    canon = {}
    for t in triples:
        for side in ("subject", "object"):
            key = (t[f"{side}_type"], t[side].strip())
            if key not in canon:
                canon[key] = resolve_name(s, key[1], key[0], facts, text)
                if canon[key][1] != "exact":
                    print(f"    {key[1]!r} ({key[0]}) -> {canon[key][0]!r}  [{canon[key][1]}]")
    for t in triples:
        st, ot = t["subject_type"], t["object_type"]
        assert st in NODES and ot in NODES  # labels/relation are pasted into Cypher: whitelist (validate did rel)
        (sn, _), (on, _) = canon[(st, t["subject"].strip())], canon[(ot, t["object"].strip())]
        for label, surface, name in ((st, t["subject"].strip(), sn), (ot, t["object"].strip(), on)):
            s.run(f"""MERGE (n:{label} {{name: $name}})
                      SET n.aliases = CASE WHEN $alias IS NULL OR $alias IN coalesce(n.aliases, [])
                                      THEN coalesce(n.aliases, []) ELSE coalesce(n.aliases, []) + $alias END""",
                  name=name, alias=surface if surface != name else None)
        # ponytail: `until` is last-writer-wins (a later doc marking history overrides 'current').
        # Ceiling: needs document dates to order conflicting claims properly.
        s.run(f"""MATCH (a:{st} {{name: $a}}), (b:{ot} {{name: $b}})
                  MERGE (a)-[e:{t['relation']}]->(b)
                  SET e.sources = CASE WHEN $doc IN coalesce(e.sources, []) THEN e.sources
                                  ELSE coalesce(e.sources, []) + $doc END,
                      e.role = coalesce(e.role, $role), e.since = coalesce(e.since, $since),
                      e.until = coalesce($until, e.until)""",
              a=sn, b=on, doc=path.name, role=t.get("role"), since=t.get("since"), until=t.get("until"))
        if t["relation"] in SINGLE_CURRENT and t.get("until") is None:
            # Cardinality: this doc asserts a CURRENT edge -> end every other current edge of the subject.
            # ended_by = which doc closed it (property-level provenance for this one property).
            # ponytail: newest-ingested wins (no doc dates); ceiling noted in 2.3.
            ended = s.run(f"""MATCH (a:{st} {{name: $a}})-[e:{t['relation']}]->(b) WHERE b.name <> $b AND e.until IS NULL
                             SET e.until = coalesce($since, date().year), e.ended_by = $doc
                             RETURN b.name AS old""", a=sn, b=on, since=t.get("since"), doc=path.name).data()
            for r in ended:
                print(f"    cardinality: {sn} {t['relation']} {r['old']} ended (superseded by {on}, {path.name})")
    for x in out.get("titles", []):
        if x.get("job_title") and (c := canon.get(("Person", x.get("name", "").strip()))):
            s.run("MATCH (p:Person {name: $n}) SET p.title = $t", n=c[0], t=x["job_title"])
    register(s, path)
    print(f"  added {path.name}: {len(triples)} facts")


def register(s, path):
    s.run("""MERGE (d:Document {name: $name})
             SET d.hash = $hash, d.version = coalesce(d.version, 0) + 1, d.ingested_at = $now""",
          name=path.name, hash=sha(path.read_text()), now=datetime.now(timezone.utc).isoformat())


def sync(s, dirs):
    on_disk = files(dirs)
    known = {r["name"]: r["hash"] for r in s.run("MATCH (d:Document) RETURN d.name AS name, d.hash AS hash")}
    added = [n for n in on_disk if n not in known]
    changed = [n for n in on_disk if n in known and known[n] != sha(on_disk[n].read_text())]
    deleted = [n for n in known if n not in on_disk]
    print(f"sync: {len(added)} new, {len(changed)} changed, {len(deleted)} deleted, "
          f"{len(on_disk) - len(added) - len(changed)} unchanged (skipped)")
    for n in deleted:
        remove(s, n)
    for n in changed:  # update = remove old version + add new
        remove(s, n)
        add(s, on_disk[n])
    for n in added:
        add(s, on_disk[n])


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    with db.session() as s:
        if cmd == "bootstrap":
            for p in files(args).values():
                register(s, p)
            print(f"registered {len(files(args))} docs")
        elif cmd == "sync":
            sync(s, args)
        elif cmd == "remove":
            remove(s, args[0])
        print("status:", s.run("MATCH (d:Document) RETURN count(d) AS docs").single()["docs"], "docs |",
              s.run("MATCH (n) WHERE NOT n:Document RETURN count(n) AS n").single()["n"], "entities |",
              s.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"], "edges")
    db.close()


if __name__ == "__main__":
    main()
