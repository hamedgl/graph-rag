"""Stage 2.2 step 1: discover what the fixed schema drops ("review bucket").

Per doc: LLM lists facts the schema can NOT express (with an evidence quote).
Then:    one LLM call groups them into proposed schema changes, with counts.
Output:  stage2/out/schema_gaps.json (raw gaps) + schema_proposals.json (grouped)

Run: .venv/bin/python stage2/discover.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "stage1"))
from extract import obj  # noqa: E402
from llm import INGEST_MODEL, call_json  # noqa: E402
from schema import as_prompt  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
DOCS = {p.name: p.read_text() for p in sorted([*(ROOT / "stage1" / "data" / "docs").glob("*.md"),
                                                *(HERE / "data" / "docs_extra").glob("*.md")])}

GAPS = f"""Our company knowledge graph uses this FIXED schema:

{as_prompt()}

List facts in the document that this schema can NOT express: a missing relation type, a missing
property on an existing relation or node, or history the 'current facts only' rule throws away.
Do NOT list facts the schema already covers. Do NOT list trivia (lunch, coffee machines, parties).
Name relations in UPPER_SNAKE_CASE. Quote the evidence verbatim. Empty list if none."""
GAP_SCHEMA = obj(gaps={"type": "array", "items": obj(
    subject={"type": "string"}, relation={"type": "string"}, object={"type": "string"},
    kind={"type": "string", "enum": ["new_relation", "new_property", "history"]},
    evidence={"type": "string"})})

GROUP = """Below are facts a knowledge-graph schema could not express, found across many documents.
Group them into a short list of proposed schema changes. Merge synonyms (e.g. HAS_ROLE / HAS_TITLE).
For each proposal give: the change (e.g. 'new relation (:Person)-[:HAS_TITLE]->(:Title)' or
'add property since to LEADS'), whether it is additive or breaking, the example facts it covers,
and the questions it would let users answer."""
GROUP_SCHEMA = obj(proposals={"type": "array", "items": obj(
    change={"type": "string"}, compatibility={"type": "string", "enum": ["additive", "breaking"]},
    examples={"type": "array", "items": {"type": "string"}},
    enables_questions={"type": "array", "items": {"type": "string"}})})


def main():
    OUT.mkdir(exist_ok=True)
    gaps = []
    for name, text in DOCS.items():
        for g in call_json(INGEST_MODEL, GAPS, text, "gaps", GAP_SCHEMA)["gaps"]:
            gaps.append({**g, "source": name})
        print(f"  {name}: {sum(g['source'] == name for g in gaps)} gaps", flush=True)
    (OUT / "schema_gaps.json").write_text(json.dumps(gaps, indent=2))

    listing = "\n".join(f"- [{g['kind']}] {g['subject']} {g['relation']} {g['object']} "
                        f"(\"{g['evidence']}\", {g['source']})" for g in gaps)
    proposals = call_json(INGEST_MODEL, GROUP, listing, "proposals", GROUP_SCHEMA)["proposals"]
    (OUT / "schema_proposals.json").write_text(json.dumps(proposals, indent=2))

    print(f"\n{len(gaps)} off-schema facts -> {len(proposals)} proposals\n")
    for i, p in enumerate(proposals, 1):
        # .get: gemini's tool calls do NOT reliably include every 'required' field (observed: 'examples')
        print(f"{i}. [{p.get('compatibility')}] {p.get('change')}  ({len(p.get('examples', []))} examples)")
        for e in p.get("examples", [])[:3]:
            print(f"     e.g. {e}")
        print(f"     enables: {'; '.join(p.get('enables_questions', [])[:2])}")


if __name__ == "__main__":
    main()
