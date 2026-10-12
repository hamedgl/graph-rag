# Toy company dataset (Stage 1.1)

- `facts.csv`: ground truth, 77 facts. The "real world" we check extraction and answers against.
- `docs/`: 20 messy documents that express those facts. This is what the pipeline ingests.

## Shape
- 15 people, 3 teams (Payments, Search, Platform), 3 offices (Toronto, Berlin, Lisbon)
- 6 projects (Atlas, Ledger, Orion, Lens, Forge, Beacon), 5 techs (Kafka, Postgres, Elasticsearch, PyTorch, Kubernetes)
- Relations: LEADS, MEMBER_OF, REPORTS_TO, LOCATED_IN, OWNS, USES, DEPENDS_ON, WORKS_ON {role, since}

## Deliberate traps (what makes extraction hard)
| Trap | Where | Correct reading |
|---|---|---|
| Name variants | 04 "Dr. Nair", 12 "A. Chen", 08 "Hiro T.", 15 first names only | Same person as full name |
| Tech aliases | 06 "PG", 05 "PostgreSQL", 09 "K8s" | Postgres, Kubernetes |
| Team alias | 05 "payments squad", 20 "search folks" | Payments, Search |
| Pronouns | 02 "He", 11 "She", 03 "He" | Dave, Lena, Omar |
| Role, not name | 04 "reports to the CTO" | Maria Santos (needs doc 01/02/17) |
| Implicit team | 02 "Dave Lee runs the team" | Team name only in the heading |
| Stale fact | 03 Omar "moved from Platform" | Current: Search only. Extracting MEMBER_OF Platform is wrong |
| Cross-team work | 11 Lena joins Atlas, stays on Platform | WORKS_ON Atlas but MEMBER_OF Platform |
| Ambiguous claim | 18 Hiro "helping with Orion experiments" | NOT a WORKS_ON fact in ground truth |
| Hub node | Kafka used by 4 of 6 projects | Causes traversal noise |
| Noise docs | 19 holiday policy, chat chatter | No facts |

## Check
`python3 stage1/check_data.py` confirms every fact is mentioned in some doc.
