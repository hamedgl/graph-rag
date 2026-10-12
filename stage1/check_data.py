"""Sanity check: every ground-truth fact is mentioned in at least one doc.

Run: python3 stage1/check_data.py
A fact counts as covered if one doc mentions both its subject and object
(by full name or a known alias). Heuristic: it proves the info is there,
not that an extractor will find it.
"""
import csv
from pathlib import Path

DATA = Path(__file__).parent / "data"

# Name variants deliberately used in the docs. Extraction must resolve these too.
ALIASES = {
    "Alice Chen": ["Alice", "A. Chen"],
    "Priya Nair": ["Priya", "Dr. Nair"],
    "Maria Santos": ["Maria", "CTO"],
    "Hiro Tanaka": ["Hiro T."],
    "Dave Lee": ["Dave"],
    "Grace Liu": ["Grace"],
    "Ines Moreau": ["Ines"],
    "Kofi Mensah": ["Kofi"],
    "Lena Fischer": ["Lena"],
    "Carol Diaz": ["Carol"],
    "Postgres": ["PostgreSQL", "PG"],
    "Kubernetes": ["K8s"],
}

docs = {p.name: p.read_text().lower() for p in sorted((DATA / "docs").glob("*.md"))}


def mentions(text, name):
    return any(n.lower() in text for n in [name, *ALIASES.get(name, [])])


facts = list(csv.DictReader(open(DATA / "facts.csv")))
missing = []
for f in facts:
    hits = [d for d, t in docs.items() if mentions(t, f["subject"]) and mentions(t, f["object"])]
    if not hits:
        missing.append(f)

print(f"{len(facts)} facts, {len(docs)} docs, {len(missing)} uncovered")
for f in missing:
    print("  MISSING:", f["subject"], f["relation"], f["object"])
assert not missing
