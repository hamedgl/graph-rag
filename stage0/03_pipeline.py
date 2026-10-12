"""Stage 0.3 demo: the 5 pipeline stages, end to end, on toy data.

Run: python3 stage0/03_pipeline.py  (stdlib only)
Extraction and answering are stubbed (regex / template) so every stage is visible.
Stage 1 replaces both stubs with OpenAI calls.
"""
import re

# 1. INGEST: raw documents (one has a pronoun, one has a name variant)
DOCS = {
    "org.txt": "Alice Chen leads Payments. Bob Kumar reports to Alice Chen.",
    "projects.txt": "Payments owns Atlas. Search owns Orion.",
    "people.txt": "Carol Diaz works on Atlas. She reports to A. Chen.",
}

# 2. EXTRACT: text -> (subject, RELATION, object, source). Schema limits relation types.
SCHEMA = {"leads": "LEADS", "reports to": "REPORTS_TO", "owns": "OWNS", "works on": "WORKS_ON"}
PATTERN = re.compile(r"([A-Z][\w.]*(?: [A-Z]\w*)?) (leads|reports to|owns|works on) ([A-Z][\w.]*(?: [A-Z]\w*)?)\.")


def extract(doc_id, text):
    # ponytail: regex stands in for the LLM; it shows the failure points, not the quality
    return [(s, SCHEMA[r], o, doc_id) for s, r, o in PATTERN.findall(text)]


# 3. STORE: edges with provenance (which doc each fact came from)
graph = [e for doc_id, text in DOCS.items() for e in extract(doc_id, text)]
print("--- Graph ---")
for s, r, o, src in graph:
    print(f"({s})-[:{r}]->({o})   from {src}")


# 4. RETRIEVE: find entities named in the question, expand 2 hops (either direction)
def retrieve(question, hops=2):
    seeds = {n for s, _, o, _ in graph for n in (s, o) if n in question}
    frontier, found = set(seeds), set()
    for _ in range(hops):
        step = {e for e in graph if e[0] in frontier or e[2] in frontier}
        found |= step
        frontier = {n for s, _, o, _ in step for n in (s, o)}
    return seeds, found


# 5. ANSWER: subgraph -> text context -> (LLM). Here we just print the prompt.
def build_prompt(question):
    seeds, edges = retrieve(question)
    facts = "\n".join(f"- {s} {r} {o} [{src}]" for s, r, o, src in sorted(edges))
    return f"Seeds: {seeds}\nFacts:\n{facts}\nQuestion: {question}"


for q in ["Who leads the team that owns Atlas?", "Who reports to Alice Chen?"]:
    print(f"\n--- {q} ---\n{build_prompt(q)}")
