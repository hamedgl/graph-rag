"""Stage 0.2 demo: same facts in two graph models + node-vs-property choice.

Run: python3 stage0/02_graph_basics.py  (stdlib only)
Fact: "Carol Diaz works on Project Atlas as backend engineer since 2025."
"""

# --- Model A: triples (RDF style). Only subject-predicate-object. ---
# "As backend engineer since 2025" describes the RELATIONSHIP, not Carol or Atlas.
# Triples can't attach data to an edge, so you invent a middle node (reification).
TRIPLES = [
    ("Carol Diaz", "hasAssignment", "assignment_1"),
    ("assignment_1", "onProject", "Project Atlas"),
    ("assignment_1", "role", "backend engineer"),
    ("assignment_1", "since", "2025"),
]

# --- Model B: labeled property graph (Neo4j style). ---
# Nodes have a label (type) + properties. Edges have a type + properties.
NODES = {
    "carol": {"label": "Person", "name": "Carol Diaz", "office": "Berlin"},
    "bob": {"label": "Person", "name": "Bob Kumar", "office": "Berlin"},
    "alice": {"label": "Person", "name": "Alice Chen", "office": "Toronto"},
    "atlas": {"label": "Project", "name": "Project Atlas"},
}
EDGES = [
    ("carol", "WORKS_ON", "atlas", {"role": "backend engineer", "since": 2025}),
]

print("Triples needed", len(TRIPLES), "statements + 1 artificial node for one fact.")
print("Property graph needed 1 edge:", EDGES[0])

# --- Node vs property: "office" stored as a property ---
# Q: "Who shares an office with Bob?" -> must scan EVERY person and compare strings.
bob_office = NODES["bob"]["office"]
same = [n["name"] for k, n in NODES.items() if n["label"] == "Person" and k != "bob" and n["office"] == bob_office]
print("\nOffice as property -> full scan:", same)

# --- Same data with office promoted to a node ---
# Now "Berlin" is a hub; the answer is: Bob -> Berlin -> everyone else at Berlin.
LOCATED_IN = [("carol", "Berlin"), ("bob", "Berlin"), ("alice", "Toronto")]
office = next(o for p, o in LOCATED_IN if p == "bob")
same2 = [NODES[p]["name"] for p, o in LOCATED_IN if o == office and p != "bob"]
print("Office as node     -> 2-hop walk:", same2)
assert same == same2 == ["Carol Diaz"]

# ponytail: on 3 people both are instant. The difference: as a node, Berlin can
# carry its own edges (Berlin -IN-> Germany, Berlin -HOSTS-> Payments team),
# so the graph can answer questions that chain through location.
