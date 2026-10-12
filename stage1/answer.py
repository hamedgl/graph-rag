"""Stage 1.6: retrieved graph context -> grounded LLM answer with citations.

Run: .venv/bin/python stage1/answer.py   (demo questions + self-checks)
"""
import json
import re

from llm import QUERY_MODEL, call_json
from retrieve import db, entity_catalog, text2cypher, traverse, vector_search

INSTRUCTIONS = """Answer the question using ONLY the facts in the context (internal company data).
- No outside knowledge, no guessing. Cite the source documents the facts came from.
- If the context does not contain the answer (or is empty), set known=false and say so briefly.
- A query result with NO ROWS means the information is not available: known=false (not 'nobody').
- Be concise: one or two sentences."""

ANSWER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "citations", "known"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "known": {"type": "boolean"},
    }}


def name_resolution(question):
    """The answer LLM sees canonical names; tell it which question terms they came from."""
    q = question.lower()
    return [f"'{alias}' = {name}" for _, name, aliases in entity_catalog() for alias in aliases
            if re.search(rf"\b{re.escape(alias.lower())}\b", q)]


def retrieve_context(question, style):
    if style == "vector":  # baseline: raw doc text, no graph, no name resolution
        hits = vector_search(question)
        context = "\n\n".join(f"[source: {name}]\n{text}" for name, _, text in hits)
        return context, "top-3: " + ", ".join(f"{name} ({score:.2f})" for name, score, _ in hits)
    if style == "cypher":
        cypher, rows, status = text2cypher(question)
        # Rows alone lose meaning: filters (e.g. REPORTS_TO Priya Nair) live in the query, not the rows.
        lines = [f"Query that produced the rows: {' '.join(cypher.split())}", *(json.dumps(r) for r in rows or [])]
        if not rows:  # a query states what it LOOKED for, not what exists (Stage 2.7: 'Mallory is a Person')
            lines.append("RESULT: NO ROWS. The graph has no matching data; nothing in the query text is a fact.")
    else:
        seeds, facts = traverse(question)
        lines = [f"- {a} {rel} {b}" + "".join(f" ({k}: {v})" for k, v in (("role", role), ("since", since),
                 ("until", until)) if v) + (" [PAST, no longer true]" if until else "") + f" [sources: {src}]"
                 for a, rel, b, role, since, src, until in facts]
        status = f"seeds={seeds}, {len(facts)} facts"
    if names := name_resolution(question):
        lines.insert(0, "Names in the question: " + "; ".join(names))
    return "\n".join(lines), status


def answer(question, style="cypher"):
    return answer_from_context(question, *retrieve_context(question, style))


def answer_from_context(question, context, status=""):
    """Shared answer step: any retriever's context -> same LLM, same rules (fair comparisons)."""
    out = call_json(QUERY_MODEL, INSTRUCTIONS, f"Context:\n{context or '(empty)'}\n\nQuestion: {question}",
                    "answer", ANSWER_SCHEMA)
    # Code check, not LLM judgement: every cited doc must appear in the retrieved context.
    in_context = set(re.findall(r"\d\d_[\w]+\.md", context))
    out["invented_citations"] = sorted(set(out["citations"]) - in_context)
    out["retrieval"] = status
    out["context"] = context  # kept so a judge can check faithfulness later (stage2/eval_scale.py)
    return out


if __name__ == "__main__":
    cases = [
        ("Who leads the team that owns Atlas?", "cypher", True),
        ("Who reports to Dr. Nair, and which office is each of them in?", "cypher", True),
        ("Who works on Atlas?", "traverse", True),            # 2-hop traversal: 36 facts, 27 off-topic
        ("Who works on Project Zeus?", "cypher", False),      # entity doesn't exist
        ("What is Alice Chen's salary?", "traverse", False),  # entity exists, fact doesn't
    ]
    for q, style, should_know in cases:
        a = answer(q, style)
        flag = "OK " if a["known"] == should_know and not a["invented_citations"] else "BAD"
        print(f"\n[{flag}] ({style}) {q}\n  retrieval: {a['retrieval']}\n  answer:    {a['answer']}"
              f"\n  known={a['known']}  citations={a['citations']}  invented={a['invented_citations']}")
    db.close()
