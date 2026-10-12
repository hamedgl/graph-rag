"""Stage 2.1: entity resolution that scales (no whole-corpus prompt).

1 MENTIONS   per doc: triples with names exactly as written ('Dr. Nair', 'A. Chen')
2 BLOCKING   candidate pairs: same type + (shared token | similar spelling | name in other's context | name embedding)
3a MATCHING  anchors (full names, non-person entities): rule merge -> rule reject -> pairwise LLM judge
3b LINKING   partial person mentions ('Hiro T.', 'Alex', 'Dr. Nair'): LLM picks ONE of all candidate anchors, or none
4 CLUSTERING union-find with cannot-link constraints
5 CANONICAL  best surface per cluster; rewrite triples
6 EVALUATE   must-merge / must-not-merge pairs + facts (one-to-one cluster->gold, so under-merges count)

All ingest LLM calls use INGEST_MODEL (.env). Run:
  .venv/bin/python stage2/resolve.py              (step 1 cached in out/raw_triples.jsonl)
  .venv/bin/python stage2/resolve.py --reextract
"""
import csv
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "stage1"))
from extract import obj  # noqa: E402
from llm import INGEST_MODEL, call_json  # noqa: E402
from schema import NODES, RELATIONS, as_prompt, validate  # noqa: E402
from score import score as prf  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "out"
DOCS = {p.name: p.read_text() for p in sorted([*(ROOT / "stage1" / "data" / "docs").glob("*.md"),
                                                *(HERE / "data" / "docs_extra").glob("*.md")])}
NAME_SIM = 0.85  # calibrated in v1: short names all embed >= 0.7; true matches 0.87-0.95
# Removed in v2: context-embedding reject. Measured: every context pair >= 0.68 (same company,
# same vocabulary), so no threshold separated matches from non-matches.


# ---------- 1 MENTIONS ----------
MENTION_INSTRUCTIONS = f"""You extract a knowledge graph from ONE internal company document.

{as_prompt()}

OVERRIDE for this task: write every name EXACTLY as it appears in this document ('Dr. Nair', 'A. Chen',
'the CTO', typos included). Only resolve pronouns (he/she/they) to the name used in this document.
A later step merges names across documents. role only for WORKS_ON; since/until only where the schema
allows them; otherwise null. Also list every person whose job title the document states (titles)."""

TRIPLES_SCHEMA = obj(triples={"type": "array", "items": obj(
    subject={"type": "string"}, subject_type={"type": "string", "enum": list(NODES)},
    relation={"type": "string", "enum": list(RELATIONS)},
    object={"type": "string"}, object_type={"type": "string", "enum": list(NODES)},
    role={"type": ["string", "null"]}, since={"type": ["integer", "null"]}, until={"type": ["integer", "null"]},
)}, titles={"type": "array", "items": obj(name={"type": "string"}, job_title={"type": "string"})})
# 'job_title', not 'title': a property literally named 'title' is silently dropped for gemini via the
# gateway (title is also a JSON-Schema annotation keyword). Stored as n.title in the graph.


def extract_raw():
    triples, titles = [], []
    for name, text in DOCS.items():
        out = call_json(INGEST_MODEL, MENTION_INSTRUCTIONS, text, "triples", TRIPLES_SCHEMA)
        for t in out["triples"]:
            props = [k for k in ("role", "since", "until") if t.get(k) is not None]
            if validate(t["subject_type"], t["relation"], t["object_type"], props) is None:
                triples.append({**t, "source": name})
        titles += [{**x, "source": name} for x in out.get("titles", [])]  # .get: gemini may omit fields
        print(f"  extracted {name}", flush=True)
    return triples, titles


def build_mentions(triples):
    # ponytail: same (type, surface) across docs = same mention. Ceiling: two different people
    # with the identical full name; fix = resolve per (doc, surface) instead.
    m = {}
    for t in triples:
        for side in ("subject", "object"):
            key = (t[f"{side}_type"], t[side].strip())
            e = m.setdefault(key, {"type": key[0], "name": key[1], "facts": [], "docs": set(), "count": 0})
            e["facts"].append(f"{t['subject']} {t['relation']} {t['object']}")
            e["docs"].add(t["source"])
            e["count"] += 1
    facts_of = defaultdict(list)  # surface name -> facts, for 1-hop neighbour context
    for e in m.values():
        facts_of[e["name"]] += e["facts"]
    for e in m.values():
        sents = [s.strip() for d in sorted(e["docs"]) for s in re.split(r"(?<=[.!?])\s+|\n", DOCS[d])
                 if e["name"].lower() in s.lower()]
        own = list(dict.fromkeys(e["facts"]))
        nbrs = {n for f in own for n in facts_of if n != e["name"] and f" {n}" in f" {f}"}
        hop = [f for n in sorted(nbrs) for f in facts_of[n] if f not in own][:8]
        # ponytail: 1-hop only; collective ER (iterate as merges change neighbours) if this proves insufficient
        e["context"] = (f"{e['name']} ({e['type']}). Facts: {'; '.join(own)}. Related facts: "
                        f"{'; '.join(dict.fromkeys(hop)) or '-'}. Text: {' | '.join(dict.fromkeys(sents))}")[:1000]
    return list(m.values())


# ---------- 2 BLOCKING ----------
def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).replace("the ", "").strip()


TITLES = {"the", "cto", "ceo", "cfo", "vp", "dr", "mr", "ms", "mrs", "head", "lead"}


def is_full_name(e):
    toks = e["name"].split()
    return (e["type"] == "Person" and len(toks) >= 2 and not re.search(r"[.()]", e["name"])
            and all(len(t) > 1 and t[0].isupper() and t.lower() not in TITLES for t in toks))


def is_partial_person(e):
    return e["type"] == "Person" and not is_full_name(e)


ALIASES = """For each name (a team, project, technology or office in a company knowledge graph), list other
names the SAME thing is commonly written as: abbreviations, full forms, common spellings
(e.g. 'K8s' <-> 'Kubernetes'). Do not list related-but-different things. Empty list if none."""
ALIAS_SCHEMA = obj(items={"type": "array", "items": obj(name={"type": "string"},
                                                        aliases={"type": "array", "items": {"type": "string"}})})


def generate_aliases(mentions):
    names = sorted({m["name"] for m in mentions if m["type"] != "Person"})
    out = call_json(INGEST_MODEL, ALIASES, "\n".join(names), "aliases", ALIAS_SCHEMA)["items"]
    return {o["name"]: {norm(a) for a in o["aliases"]} for o in out}


def block(mentions, name_sim, aliases):
    pairs = []
    for i, j in combinations(range(len(mentions)), 2):
        a, b = mentions[i], mentions[j]
        if a["type"] != b["type"]:
            continue
        na, nb = norm(a["name"]), norm(b["name"])
        if (set(na.split()) & set(nb.split())
                or difflib.SequenceMatcher(None, na, nb).ratio() > 0.75
                or re.search(rf"\b{re.escape(na)}\b", norm(b["context"]))
                or re.search(rf"\b{re.escape(nb)}\b", norm(a["context"]))
                or name_sim[i, j] > NAME_SIM
                or nb in aliases.get(a["name"], ()) or na in aliases.get(b["name"], ())):
            pairs.append((i, j))
    return pairs


# ---------- 3a MATCHING (anchors, pairwise) ----------
JUDGE = """Do these two mentions from internal company documents refer to the SAME real-world entity?
Abbreviations, aliases and typos can refer to the same entity.
If the evidence is insufficient or ambiguous, answer same=false: a wrong merge is worse than a missed one."""
JUDGE_SCHEMA = obj(same={"type": "boolean"}, reason={"type": "string"})


def match(a, b):
    if norm(a["name"]) == norm(b["name"]):
        return True, "rule: same normalized name"
    if is_full_name(a) and is_full_name(b) and difflib.SequenceMatcher(
            None, norm(a["name"]), norm(b["name"])).ratio() < 0.85:
        return False, "rule: different full names"
    out = call_json(INGEST_MODEL, JUDGE, f"Mention A: {a['context']}\n\nMention B: {b['context']}", "judge", JUDGE_SCHEMA)
    return out["same"], f"llm: {out['reason']}"


# ---------- 3b LINKING (partial person mentions vs ALL candidate anchors) ----------
LINK = """A name from an internal company document may refer to an entity already known under another name.
The candidates are ALL known entities of the same type whose names could match it (closed world).
- If exactly one candidate is the same entity (alias, abbreviation, initials, first name, title, typo)
  and the context does not contradict it, choose it. Use the related facts: they connect entities.
- If no candidate is the same entity, or several fit and the context does not clearly pick one, choose 0.
Answer with the candidate number, or 0 for none."""
LINK_SCHEMA = obj(choice={"type": "integer"}, reason={"type": "string"})


def link(mention, candidates):
    listing = "\n".join(f"{k}. {c['context']}" for k, c in enumerate(candidates, 1))
    out = call_json(INGEST_MODEL, LINK, f"Mention: {mention['context']}\n\nCandidates:\n{listing}", "link", LINK_SCHEMA)
    k = out["choice"]
    return (k - 1 if 1 <= k <= len(candidates) else None), f"llm-link: {out['reason']}"


# ---------- 4 CLUSTERING with cannot-link ----------
def cluster(n, merges, cannot):
    parent = list(range(n))
    members = {i: {i} for i in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    blocked = []
    for i, j in merges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if any((x, y) in cannot or (y, x) in cannot for x in members[ri] for y in members[rj]):
            blocked.append((i, j))  # merging would fuse two known-different entities
            continue
        parent[rj] = ri
        members[ri] |= members.pop(rj)
    return [sorted(m) for m in members.values()], blocked


# ---------- 5 CANONICAL ----------
def canonical(group):
    # prefer: no initials/dots, full name for people, most mentions, then longest
    return max(group, key=lambda e: ("." not in e["name"], is_full_name(e), e["count"], len(e["name"])))["name"]


def resolve(mentions, pairs):
    decisions, rule_merges, llm_merges, cannot = [], [], [], set()
    candidates = defaultdict(list)
    for i, j in pairs:
        a, b = mentions[i], mentions[j]
        if a["type"] != "Person" and norm(a["name"]) != norm(b["name"]) and a["count"] != b["count"]:
            weak, strong = (i, j) if a["count"] < b["count"] else (j, i)
            candidates[weak].append(strong)  # link the rarer name into the more established one
            continue
        if is_partial_person(a) or is_partial_person(b):
            if is_partial_person(a) != is_partial_person(b):  # partial vs anchor -> linking step
                p, anchor = (i, j) if is_partial_person(a) else (j, i)
                candidates[p].append(anchor)
            continue  # partial vs partial: resolved through their anchors
        same, why = match(a, b)
        decisions.append({"a": a["name"], "b": b["name"], "type": a["type"], "same": same, "why": why})
        if why.startswith("llm"):
            print(f"  pair {a['name']} / {b['name']}: {'SAME' if same else 'diff'}", flush=True)
        if same:
            (rule_merges if why.startswith("rule") else llm_merges).append((i, j))
        elif why.startswith("rule"):
            cannot.add((i, j))  # only certain rejections become hard constraints

    for p, cands in candidates.items():
        k, why = link(mentions[p], [mentions[c] for c in cands])
        chosen = mentions[cands[k]]["name"] if k is not None else None
        decisions.append({"a": mentions[p]["name"], "b": chosen, "candidates": [mentions[c]["name"] for c in cands],
                          "type": mentions[p]["type"], "same": k is not None, "why": why})
        print(f"  link {mentions[p]['name']!r} -> {chosen!r} (from {[mentions[c]['name'] for c in cands]})", flush=True)
        if k is not None:
            llm_merges.append((p, cands[k]))  # each partial joins at most ONE anchor: no chaining
    return decisions, rule_merges + llm_merges, cannot  # certain merges first


def write_entities(raw, resolved, path, titles=None):
    """entities.json in stage1 format: canonical name, type, aliases (= raw surfaces merged into it), title."""
    ents = {}
    for r, t in zip(raw, resolved):  # same order: resolved = raw with names rewritten
        for side in ("subject", "object"):
            e = ents.setdefault(t[side], {"name": t[side], "type": t[f"{side}_type"], "aliases": set()})
            if r[side].strip() != t[side]:
                e["aliases"].add(r[side].strip())
    for name, title in (titles or {}).items():
        if name in ents:
            ents[name]["title"] = title
    path.write_text(json.dumps([{**e, "aliases": sorted(e["aliases"])} for e in ents.values()], indent=2))
    return len(ents)


def main():
    OUT.mkdir(exist_ok=True)
    raw_path, titles_path = OUT / "raw_triples.jsonl", OUT / "raw_titles.json"
    if "--reextract" in sys.argv or not raw_path.exists():
        raw, raw_titles = extract_raw()
        raw_path.write_text("".join(json.dumps(t) + "\n" for t in raw))
        titles_path.write_text(json.dumps(raw_titles, indent=2))
    triples = [json.loads(line) for line in open(raw_path)]
    raw_titles = json.loads(titles_path.read_text()) if titles_path.exists() else []
    mentions = build_mentions(triples)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("google/embeddinggemma-2")
    name_emb = model.encode([m["name"] for m in mentions], normalize_embeddings=True)
    aliases = generate_aliases(mentions)
    print(f"  aliases: { {k: sorted(v) for k, v in aliases.items() if v} }", flush=True)
    pairs = block(mentions, name_emb @ name_emb.T, aliases)
    decisions, merges, cannot = resolve(mentions, pairs)
    groups, blocked = cluster(len(mentions), merges, cannot)

    canon, size = {}, {}
    for g in groups:
        name = canonical([mentions[i] for i in g])
        size[name] = sum(mentions[i]["count"] for i in g)
        for i in g:
            canon[(mentions[i]["type"], mentions[i]["name"])] = name
    resolved = [{**t, "subject": canon[(t["subject_type"], t["subject"].strip())],
                 "object": canon[(t["object_type"], t["object"].strip())]} for t in triples]
    (OUT / "triples.jsonl").write_text("".join(json.dumps(t) + "\n" for t in resolved))
    (OUT / "er_decisions.json").write_text(json.dumps(decisions, indent=2))
    surface_to_canon = {name: c for (typ, name), c in canon.items() if typ == "Person"}
    titles = {}  # canonical person -> most frequent stated title
    for t in raw_titles:  # .get: gemini sometimes omits 'required' fields
        if t.get("job_title") and (c := surface_to_canon.get(t.get("name", "").strip())):
            titles.setdefault(c, Counter())[t["job_title"]] += 1
    titles = {c: cnt.most_common(1)[0][0] for c, cnt in titles.items()}
    print(f"titles: {titles}")
    write_entities(triples, resolved, OUT / "entities.json", titles)

    # ---------- 6 EVALUATE ----------
    gold = json.loads((HERE / "data" / "er_pairs.json").read_text())
    stage = Counter(d["why"].split(":")[0] + (" merge" if d["same"] else " reject") for d in decisions)
    print(f"\ningest model {INGEST_MODEL} | mentions {len(mentions)} | candidate pairs {len(pairs)} "
          f"(of {len(mentions) * (len(mentions) - 1) // 2}) | decisions {dict(stage)}")
    print(f"entities after resolution: {len(groups)} (truth 38) | merges blocked by cannot-link: "
          f"{[(mentions[i]['name'], mentions[j]['name']) for i, j in blocked]}")

    cluster_of = {mentions[i]["name"].lower(): k for k, g in enumerate(groups) for i in g}
    for label, gold_pairs, want in (("must merge", gold["must_merge"], True),
                                    ("must NOT merge", gold["must_not_merge"], False)):
        res = [(a, b, None if a.lower() not in cluster_of or b.lower() not in cluster_of
                else cluster_of[a.lower()] == cluster_of[b.lower()]) for a, b in gold_pairs]
        seen = [r for r in res if r[2] is not None]
        print(f"{label:15} {sum(r[2] == want for r in seen)}/{len(seen)} correct "
              f"({len(res) - len(seen)} pairs not seen as mentions)")
        for a, b, r in seen:
            if r != want:
                print(f"    ✗ {a} / {b}")

    # One-to-one: each gold entity is credited to its LARGEST cluster only. A leftover cluster
    # ('Dr. Nair' still separate) keeps its own name, so its facts count as wrong + missed.
    alias = {n.lower(): c for c, n in gold["must_merge"]}
    alias.update({a.lower(): c for c, al in gold["extra_aliases"].items() for a in al})
    best = {}
    for name, n in size.items():
        g_ = alias.get(name.lower(), name)
        if n > best.get(g_, (0, None))[0]:
            best[g_] = (n, name)
    to_gold = {name: (g_ if best[g_][1] == name else f"{name} [unmerged]")
               for name in size for g_ in [alias.get(name.lower(), name)]}
    truth = {(f["subject"], f["relation"], f["object"]) for p in (ROOT / "stage1" / "data" / "facts.csv",
             HERE / "data" / "facts_extra.csv", HERE / "data" / "facts_v2.csv") for f in csv.DictReader(open(p))}
    pred = {(to_gold[t["subject"]], t["relation"], to_gold[t["object"]]) for t in resolved}
    precision, recall, wrong, missed = prf(truth, pred)
    print(f"facts: precision {precision:.0%} | recall {recall:.0%} (truth {len(truth)}, predicted {len(pred)})")
    for label, items in (("WRONG", wrong), ("MISSED", missed)):
        print(f"  {label} {len(items)}: " + "; ".join(f"{s} {r} {o}" for s, r, o in sorted(items, key=lambda x: x[1])))


if __name__ == "__main__":
    assert cluster(3, [(0, 2), (1, 2)], {(0, 1)})[1] == [(1, 2)]  # chain 0-2-1 blocked: 0 and 1 cannot link
    main()
