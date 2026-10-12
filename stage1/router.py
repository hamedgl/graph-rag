"""Stage 2.4: router = classify the question, then fall back if the first retriever doesn't know.

  1. LLM ranks the 3 retrievers for this question (one small call, query model)
  2. answer with the top one; if known=false, try the next (max 3)

Run: .venv/bin/python stage1/router.py "question"   (or via eval.py --systems ... router)
"""
import sys

from answer import answer
from extract import obj
from llm import QUERY_MODEL, call_json
from schema import as_prompt

STYLES = ["cypher", "traverse", "vector"]
ROUTE = f"""Rank three retrievers for answering a question about a company. Best first.

- cypher: writes a graph query over this schema. Best for exact facts about people, teams, projects,
  technologies and offices; counts, filters, comparisons, multi-hop chains, current vs past membership.
{as_prompt()}
- traverse: walks the graph around entities NAMED in the question. Good for open "tell me about X"
  questions; useless when the question names no entity.
- vector: searches the raw document text. Best for anything the schema does NOT store: descriptions,
  purposes, reasons, events/incidents, policies, opinions."""
ROUTE_SCHEMA = obj(order={"type": "array", "items": {"type": "string", "enum": STYLES}}, reason={"type": "string"})


def route(question):
    out = call_json(QUERY_MODEL, ROUTE, question, "route", ROUTE_SCHEMA)
    order = [s for s in dict.fromkeys(out.get("order", [])) if s in STYLES]
    return order + [s for s in STYLES if s not in order], out.get("reason", "")  # always a full ranking


def routed_answer(question):
    order, reason = route(question)
    tried = []
    for style in order:
        a = answer(question, style)
        tried.append(style)
        if a["known"]:
            break
        # (Tried: stop after 2 unknowns. Reverted: it blocked misrouted descriptive questions from
        # reaching vector. The real fix for 'not found' is the schema property check in text2cypher.)
    # ponytail: falls back only on known=false. A confidently WRONG first answer is not caught;
    # that needs an answer-quality check (Stage 2.6).
    a["retrieval"] = f"route {order} -> used {tried[-1]} (tried {tried}); {a['retrieval']}"
    a["route"], a["tried"] = order, tried
    return a


if __name__ == "__main__":
    print(routed_answer(" ".join(sys.argv[1:])))
