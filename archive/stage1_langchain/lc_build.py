"""LangChain version of extract.py + load.py, two-pass.

Pass 1: LCEL chain -> canonical entity list (same instructions as stage1).
Pass 2: LLMGraphTransformer per doc, entity catalog in additional_instructions.
Then:   canonicalize in code (LangChain can't enum-lock names) -> Neo4jGraph.add_graph_documents.

Run: .venv/bin/python stage1_langchain/lc_build.py
Then: .venv/bin/python stage1_langchain/lc_build.py --score   (free; reuses stage1/score.py)
Browse: http://localhost:7475
"""
import json
import sys
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j.graphs.graph_document import Node
from pydantic import BaseModel

from lc_common import DATA, NODES, OUT, RELATIONS, RULES, graph, llm  # first: puts stage1/ on sys.path
from extract import ENTITY_INSTRUCTIONS  # noqa: E402  stage1 pass-1 prompt text
from schema import validate  # noqa: E402


class Entity(BaseModel):
    name: str
    type: Literal[tuple(NODES)]
    aliases: list[str]


class EntityList(BaseModel):
    entities: list[Entity]


def find_entities(docs):
    # Instructions go in as a variable: the schema text contains {role, since}, which a
    # template would try to fill as a placeholder.
    chain = (ChatPromptTemplate.from_messages([("system", "{instructions}"), ("human", "{corpus}")])
             | llm.with_structured_output(EntityList, method="json_schema", strict=True))
    corpus = "\n\n".join(f"=== {d.metadata['source']} ===\n{d.page_content}" for d in docs)
    return [e.model_dump() for e in chain.invoke({"instructions": ENTITY_INSTRUCTIONS, "corpus": corpus}).entities]


def canonicalize(graph_docs, entities):
    """Map every name/alias (any case: the transformer title-cases ids) to its canonical entity.
    Drop edges with unknown entities or wrong direction. Returns dropped edges."""
    lookup = {n.lower(): Node(id=e["name"], type=e["type"]) for e in entities for n in [e["name"], *e["aliases"]]}
    dropped = []
    for gd in graph_docs:
        kept = []
        for r in gd.relationships:
            s, o = lookup.get(r.source.id.lower()), lookup.get(r.target.id.lower())
            err = "unknown entity" if not (s and o) else validate(s.type, r.type, o.type, list(r.properties))
            if err:
                dropped.append((r.source.id, r.type, r.target.id, err))
                continue
            r.source, r.target = s, o
            kept.append(r)
        gd.relationships = kept
        gd.nodes = list({(n.id, n.type): n for r in kept for n in (r.source, r.target)}.values())
    return dropped


def build():
    docs = [Document(page_content=p.read_text(), metadata={"source": p.name})
            for p in sorted((DATA / "docs").glob("*.md"))]

    entities = find_entities(docs)
    (OUT / "entities.json").write_text(json.dumps(entities, indent=2))
    catalog = "\n".join(f"- {e['name']} ({e['type']}), also: {', '.join(e['aliases']) or '-'}" for e in entities)
    print(f"pass 1: {len(entities)} entities")

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=list(NODES),
        allowed_relationships=[(src, rel, dst) for rel, (src, dst, _, _) in RELATIONS.items()],
        strict_mode=True,  # drops off-schema relation types/labels (names are NOT checked)
        relationship_properties=["role", "since"],
        additional_instructions="\n".join(RULES) + f"\nUse ONLY these canonical entity ids:\n{catalog}",
    )
    graph_docs = transformer.convert_to_graph_documents(docs)  # pass 2: one LLM call per doc
    dropped = canonicalize(graph_docs, entities)
    print(f"pass 2: canonicalize dropped {len(dropped)} edges")
    for d in dropped:
        print("   DROP", d)

    # Same triple format as stage1/out/triples.jsonl, so stage1's scorer works unchanged.
    with open(OUT / "triples.jsonl", "w") as f:
        for gd in graph_docs:
            for r in gd.relationships:
                f.write(json.dumps({"subject": r.source.id, "subject_type": r.source.type, "relation": r.type,
                                    "object": r.target.id, "object_type": r.target.type,
                                    "source": gd.source.metadata["source"]}) + "\n")

    g = graph(read_only=False)
    g.query("MATCH (n) DETACH DELETE n")  # full reload
    g.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)
    nodes = g.query("MATCH (n:__Entity__) RETURN count(n) AS n")[0]["n"]
    rels = g.query("MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) AS n")[0]["n"]
    print(f"{len(docs)} docs -> {nodes} entity nodes, {rels} entity edges (+ MENTIONS edges to Document nodes)")


def score():
    import csv
    from score import score as prf  # stage1/score.py
    truth = {(f["subject"], f["relation"], f["object"]) for f in csv.DictReader(open(DATA / "facts.csv"))}
    pred = {(t["subject"], t["relation"], t["object"]) for t in map(json.loads, open(OUT / "triples.jsonl"))}
    precision, recall, wrong, missed = prf(truth, pred)
    print(f"truth {len(truth)} | extracted {len(pred)} unique | precision {precision:.0%} | recall {recall:.0%}")
    for label, items in (("WRONG", wrong), ("MISSED", missed)):
        print(f"\n{label} - {len(items)}:")
        for t in sorted(items, key=lambda x: x[1]):
            print("  ", t)


if __name__ == "__main__":
    score() if "--score" in sys.argv else build()
