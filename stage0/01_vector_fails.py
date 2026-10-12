"""Stage 0.1 demo: why similarity search misses multi-hop answers.

Run: python3 stage0/01_vector_fails.py  (stdlib only, no API key)
"""
import math
import re
from collections import Counter

# Each string = one chunk in a vector store. Facts are scattered across chunks.
CHUNKS = [
    "Alice Chen leads the Payments team.",
    "Bob Kumar reports to Alice Chen.",
    "Project Atlas is owned by the Payments team.",
    "Carol Diaz works on Project Atlas as a backend engineer.",
    "Project Atlas launched its first beta in March.",
    "Dave Lee leads the Search team, which owns Project Orion.",
]

QUESTION = "Who is the leader responsible for Project Atlas?"


# ponytail: bag-of-words cosine stands in for embeddings. Real embeddings are
# smarter about synonyms, but the structural failure shown here is the same.
def vec(text):
    return Counter(re.findall(r"[a-z]+", text.lower()))


def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a)
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


print(f"Q: {QUESTION}\n\n--- Vector RAG: top-3 chunks by similarity ---")
q = vec(QUESTION)
ranked = sorted(CHUNKS, key=lambda c: cosine(q, vec(c)), reverse=True)
for c in ranked:
    marker = "  <- retrieved" if c in ranked[:3] else ""
    print(f"{cosine(q, vec(c)):.2f}  {c}{marker}")
print("\nThe chunk that holds the answer ('Alice Chen leads the Payments team.') "
      "does not mention Atlas, so it ranks low.")

# Same facts, stored as a graph: (subject, relation, object)
EDGES = [
    ("Alice Chen", "LEADS", "Payments"),
    ("Bob Kumar", "REPORTS_TO", "Alice Chen"),
    ("Payments", "OWNS", "Project Atlas"),
    ("Carol Diaz", "WORKS_ON", "Project Atlas"),
    ("Dave Lee", "LEADS", "Search"),
    ("Search", "OWNS", "Project Orion"),
]

print("\n--- Graph RAG: follow edges ---")
owner = next(s for s, r, o in EDGES if r == "OWNS" and o == "Project Atlas")
leader = next(s for s, r, o in EDGES if r == "LEADS" and o == owner)
print(f"Project Atlas <-OWNS- {owner} <-LEADS- {leader}")
print(f"Answer: {leader}")
assert leader == "Alice Chen"
