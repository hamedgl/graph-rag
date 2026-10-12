"""Stage 1.3: docs -> triples, two passes, validated by the schema.

Pass 1: one call over ALL docs -> canonical entity list with aliases (out/entities.json).
Pass 2: per doc -> triples; subject/object restricted (enum) to canonical names.

Run: .venv/bin/python stage1/extract.py
Reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL from .env (any OpenAI-compatible endpoint).
Writes stage1/out/triples.jsonl (one triple per line, with source doc = provenance).
"""
import json
import os
from pathlib import Path

from openai import OpenAI

import env  # noqa: F401  (loads .env)

from schema import NODES, RELATIONS, as_prompt, validate

HERE = Path(__file__).parent
MODEL = os.environ["LLM_MODEL"]


def json_format(name, schema):
    return {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}


def obj(**props):
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


# --- Pass 1: entities across the whole corpus ---
ENTITY_INSTRUCTIONS = f"""You build the entity list for a company knowledge graph.

{as_prompt()}

List every entity in the documents that fits a node label. Merge all mentions of the
same real entity (initials, titles, first names, roles like 'the CTO', aliases like 'PG')
under ONE canonical full name, and list the variants as aliases."""

ENTITY_FORMAT = json_format("entities", obj(entities={"type": "array", "items": obj(
    name={"type": "string"},
    type={"type": "string", "enum": list(NODES)},
    aliases={"type": "array", "items": {"type": "string"}},
)}))


# ponytail: whole corpus in one call; ceiling = context window. At scale: per-doc entity
# pass + merge step (that merge is entity resolution, Stage 2.1).
def find_entities(client, docs):
    corpus = "\n\n".join(f"=== {name} ===\n{text}" for name, text in docs.items())
    resp = client.responses.create(model=MODEL, instructions=ENTITY_INSTRUCTIONS, input=corpus,
                                   text=ENTITY_FORMAT)
    return json.loads(resp.output_text)["entities"]


# --- Pass 2: triples per doc, names locked to the entity list ---
def extract(client, text, entities):
    names = sorted({e["name"] for e in entities})
    catalog = "\n".join(f"- {e['name']} ({e['type']}), also: {', '.join(e['aliases']) or '-'}" for e in entities)
    instructions = f"""You extract a knowledge graph from one internal company document.

{as_prompt()}

Known entities (use ONLY these canonical names for subject/object):
{catalog}

Return every fact in the document that fits the schema.
Set role/since only for WORKS_ON; otherwise null. No facts -> empty list."""
    fmt = json_format("triples", obj(triples={"type": "array", "items": obj(
        subject={"type": "string", "enum": names},
        subject_type={"type": "string", "enum": list(NODES)},
        relation={"type": "string", "enum": list(RELATIONS)},  # enums: off-schema output impossible
        object={"type": "string", "enum": names},
        object_type={"type": "string", "enum": list(NODES)},
        role={"type": ["string", "null"]},
        since={"type": ["integer", "null"]},
    )}))
    resp = client.responses.create(model=MODEL, instructions=instructions, input=text, text=fmt)
    return json.loads(resp.output_text)["triples"]


def main():
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    docs = {p.name: p.read_text() for p in sorted((HERE / "data" / "docs").glob("*.md"))}

    entities = find_entities(client, docs)
    (out / "entities.json").write_text(json.dumps(entities, indent=2))
    print(f"pass 1: {len(entities)} entities -> {out / 'entities.json'}")

    kept = dropped = 0
    with open(out / "triples.jsonl", "w") as f:
        for name, text in docs.items():
            for t in extract(client, text, entities):
                props = [k for k in ("role", "since") if t[k] is not None]
                if err := validate(t["subject_type"], t["relation"], t["object_type"], props):
                    dropped += 1
                    print(f"  DROP {name}: {t['subject']} {t['relation']} {t['object']} ({err})")
                    continue
                kept += 1
                f.write(json.dumps({**t, "source": name}) + "\n")
            print(f"{name}: done")
    print(f"\nkept {kept}, dropped {dropped} -> {out / 'triples.jsonl'}")


if __name__ == "__main__":
    main()
