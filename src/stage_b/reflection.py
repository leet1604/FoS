import json, re

SYSTEM_REFLECT = """You are a strategy critic in a molecular selectivity-optimization loop.
The search has STALLED. Do NOT retry the same kind of move. Diagnose why it stalled and
shift to a DIFFERENT transform FAMILY that has not been exhausted.
Output ONLY JSON, no prose, no markdown:
{"diagnosis":"<one line: why stalled>",
 "next_family":"<one family name to try next, from families_not_yet_tried>",
 "avoid_families":["<families already exhausted>"]}"""

def build_reflect_prompt(history_rows, tried_families, unused_families, critic_note):
    tried = "\n".join(f'  - {h["id"]} (family={h["family"]}): delta_S={h["delta_S"]:+.3f} {h["reliability"]}'
                      for h in history_rows)
    return f"""stall_reason: {critic_note}

moves_tried_so_far:
{tried}

families_already_used: {sorted(tried_families)}
families_not_yet_tried: {sorted(unused_families)}

Diagnose and pick ONE family from families_not_yet_tried. JSON only."""

def parse_reflection(raw, valid_families):
    if not raw: return None
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m: return None
    try: data = json.loads(m.group(0))
    except Exception: return None
    nf = data.get("next_family")
    return {"diagnosis": str(data.get("diagnosis",""))[:200],
            "next_family": nf if nf in valid_families else None,
            "avoid_families": [f for f in data.get("avoid_families",[]) if f in valid_families]}

def next_menu(full_menu, avoid_families, next_family=None):
    m = [x for x in full_menu if x["family"] not in avoid_families]
    if next_family:
        pref = [x for x in m if x["family"] == next_family]
        return pref if pref else m
    return m
