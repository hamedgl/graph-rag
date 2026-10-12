"""Stage 1.2: graph schema = contract between extraction and retrieval.

Run: python3 stage1/schema.py  (prints the prompt text + validates ground truth)
Used by: extraction (1.3) to constrain/validate triples, Text2Cypher (1.5) to describe the graph.

Version history (additive changes only; old queries keep working):
  v1  Stage 1 schema, current facts only
  v2  2026-10-12 (Stage 2.2): Person.title; since/until on LEADS + MEMBER_OF (history kept, until=null = current)
  v3  2026-10-12 (Stage 2.4 fix): since/until on REPORTS_TO; SINGLE_CURRENT cardinality (one current manager)
  v4  2026-10-12 (Stage 2.5): Person.employee_id + email (from HR); edge meta overridden_by (losing claim in a
      source conflict is kept but flagged)
"""
import csv
from pathlib import Path

# Node labels. Every node is identified by `name` (the key property).
NODES = {
    "Person": "An employee, by full canonical name (e.g. 'Alice Chen', never 'A. Chen' or 'she').",
    "Team": "An engineering team: Payments, Search or Platform.",
    "Project": "A software system/service the company builds, e.g. Atlas.",
    "Tech": "A third-party technology, canonical name (Postgres not PG/PostgreSQL, Kubernetes not K8s).",
    "Office": "A city where people work.",
}

SCHEMA_VERSION = 4

# Node properties beyond `name`: label -> {property: description}
NODE_PROPS = {
    "Person": {"title": "Job title if the text states one (e.g. 'CTO'). Not a project role.",
               "employee_id": "HR employee id (e.g. 'E005'), from the HR system only.",
               "email": "Work email, from the HR system only."},
}

# Relation type -> (from label, to label, allowed edge properties, description)
RELATIONS = {
    "LEADS": ("Person", "Team", ["since", "until"], "Person is/was the head/lead of the team."),
    "MEMBER_OF": ("Person", "Team", ["since", "until"], "Person belongs/belonged to the team (not the lead)."),
    "REPORTS_TO": ("Person", "Person", ["since", "until"], "Direct manager relationship (current or past)."),
    "LOCATED_IN": ("Person", "Office", [], "Office the person works from."),
    "OWNS": ("Team", "Project", [], "Team is responsible for the project."),
    "USES": ("Project", "Tech", [], "Project is built on / depends on the technology."),
    "DEPENDS_ON": ("Project", "Project", [], "Project needs another project to function."),
    "WORKS_ON": ("Person", "Project", ["role", "since"], "Person is formally assigned to the project."),
}

# Relations where a subject has at most ONE current (until IS NULL) edge. Enforced at ingest:
# a new current edge ends the previous one. (A person has one manager; MEMBER_OF is left
# multi-valued on purpose: cross-team membership is plausible.)
SINGLE_CURRENT = {"REPORTS_TO"}

# Properties any edge may carry besides its schema props (set by loaders, not by extraction)
EDGE_META = {"sources": "Documents/systems asserting the fact.",
             "overridden_by": "Set when a more authoritative source contradicts this edge (the source's name)."}

RULES = [
    "LEADS/MEMBER_OF/REPORTS_TO keep history: a past state gets until=<year> ('moved from X in 2024' -> "
    "MEMBER_OF X until 2024); current ones have until=null. All other relations: current facts only.",
    "Queries about the present must filter with WHERE r.until IS NULL on LEADS/MEMBER_OF/REPORTS_TO. "
    "Never write {until: null} inside a pattern: in Cypher it matches nothing.",
    "Edges with overridden_by set were overruled by a more authoritative source: exclude them with "
    "WHERE r.overridden_by IS NULL, unless the question asks about conflicts or disagreements.",
    "Resolve pronouns and titles to full names ('she', 'the CTO', 'Dr. Nair' -> full name).",
    "Use only the labels and relations above. Drop anything that does not fit.",
    "Informal help or one-off activity is not WORKS_ON; only formal assignment is.",
]


def as_prompt():
    """Schema rendered as plain text for LLM prompts."""
    lines = ["Node labels (key property: name):"]
    for label, desc in NODES.items():
        props = "".join(f" Property {k}: {v}" for k, v in NODE_PROPS.get(label, {}).items())
        lines.append(f"- {label}: {desc}{props}")
    lines.append("Relations:")
    for rel, (src, dst, props, desc) in RELATIONS.items():
        p = f" {{{', '.join(props)}}}" if props else ""
        lines.append(f"- (:{src})-[:{rel}{p}]->(:{dst}): {desc}")
    lines.append("Rules:")
    lines += [f"- {r}" for r in RULES]
    return "\n".join(lines)


def known_properties():
    """Every property the graph can hold (for checking generated Cypher)."""
    return ({"name", "aliases"} | set(EDGE_META) | {p for props in NODE_PROPS.values() for p in props}
            | {p for _, _, props, _ in RELATIONS.values() for p in props})


def validate(subject_type, relation, object_type, props=()):
    """Return None if the triple fits the schema, else the reason it doesn't."""
    if relation not in RELATIONS:
        return f"unknown relation {relation}"
    src, dst, allowed, _ = RELATIONS[relation]
    if (subject_type, object_type) != (src, dst):
        return f"{relation} must be {src}->{dst}, got {subject_type}->{object_type}"
    extra = set(props) - set(allowed)
    return f"unexpected properties {extra}" if extra else None


if __name__ == "__main__":
    print(as_prompt())
    facts = csv.DictReader(open(Path(__file__).parent / "data" / "facts.csv"))
    errors = []
    for f in facts:
        props = [kv.split("=")[0] for kv in f["properties"].split(";") if kv]
        if err := validate(f["subject_type"], f["relation"], f["object_type"], props):
            errors.append((f["subject"], f["relation"], f["object"], err))
    print(f"\nGround truth vs schema: {len(errors)} violations")
    assert not errors, errors
    # Negative checks: the validator must reject off-schema triples.
    assert validate("Team", "LEADS", "Person")  # wrong direction
    assert validate("Person", "MANAGES", "Team")  # synonym, not in schema
    assert validate("Person", "WORKS_ON", "Project", ["hours"])  # unknown property
