import json, re

SYSTEM = """You are a move selector in a molecular selectivity-optimization loop.
You are given a MENU of pre-validated transforms (each with an id). You do NOT invent
transforms or write SMILES/SMARTS. You ONLY pick ids from the menu.
Pick the transforms most likely to improve selectivity (delta_S) while keeping products
near measured chemical space. Output ONLY JSON, no prose, no markdown:
{"picks":[{"id":"<menu id>","reason":"<short>"}]}"""

def build_user_prompt(current_smiles, menu, scorer_hint, n=3):
    menu_str = "\n".join(f'- id="{m["id"]}" : {m["label"]} | {m["rationale"]}' for m in menu)
    return f"""current_molecule: {current_smiles}
scorer_feedback: {scorer_hint}

MENU (pick ids from here only):
{menu_str}

Pick up to {n} ids you'd try next. JSON only."""

def parse_picks(raw, valid_ids):
    if not raw: return []
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return []
    try: data = json.loads(m.group(0))
    except Exception: return []
    picks, seen = [], set()
    for p in data.get("picks", []):
        if isinstance(p, dict) and p.get("id") in valid_ids and p.get("id") not in seen:
            seen.add(p["id"]); picks.append({"id": p["id"], "reason": str(p.get("reason",""))[:80]})
    return picks
