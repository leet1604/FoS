from __future__ import annotations

DOMAIN = """DOMAIN FACTS (obey strictly):
- ΔS (delta_S) = SELECTIVITY difference = on-target binding minus off-target binding. Higher ΔS = MORE selective. It is NOT thermodynamic entropy.
- Do NOT add chemical claims (polarity, solubility, binding mode, potency) unless they appear in the given facts. If unsure why a move helps, cite only the provided evidence/rationale."""

SYSTEM_STEP = f"""You explain ONE step of a molecular selectivity-optimization run to a medicinal chemist.
{DOMAIN}
Use ONLY the facts given. NEVER invent molecules, numbers, or effects.
Write 1-2 factual sentences. No preamble."""

SYSTEM_TRAJ = f"""You summarize a molecular selectivity-optimization trajectory factually.
{DOMAIN}
Use ONLY the facts given. Write 3-4 sentences: what improved (in selectivity terms), the logic of the path, and why it stopped."""

def build_step_prompt(fact):
    ev = ""
    if fact.get("evidence_smiles"):
        ev = f' Evidence: this exact transform was observed in a MEASURED analog with selectivity change {fact["observed_delta_S"]:+.2f}.'
    rat = f'\nknown_rationale_for_this_move: {fact["rationale"]}' if fact.get("rationale") else ""
    return f"""current_molecule: {fact["current"]}
action: {fact["label"]} (family={fact["family"]}, type={fact["move_type"]}).{ev}{rat}
scored_result: predicted ΔS={fact["delta_S"]} (selectivity change), reliability={fact["reliability"]}, tier={fact["tier"]}
decision: {fact["decision"]} — {fact["decision_reason"]}
Explain in 1-2 sentences why this step helps or hurts SELECTIVITY, using only the facts above."""

def build_traj_prompt(seed_S, final_S, accepted, rejected_families, stop_reason):
    acc = "; ".join(f'{a["label"]} (ΔS {a["delta_S"]:+.2f} selectivity, {a["reliability"]})' for a in accepted) or "none"
    return f"""seed selectivity_S = {seed_S}
accepted moves in order: {acc}
final selectivity_S = {final_S}
families tried but exhausted: {sorted(rejected_families)}
stopped because: {stop_reason}
Summarize the SELECTIVITY optimization logic in 3-4 sentences."""

def _rel_of(s):
    if not s.accepted_id:
        return next((c[2] for c in s.candidates), "n/a") if s.candidates else "n/a"
    return next((c[2] for c in s.candidates if c[0]==s.accepted_id), "n/a")

def narrate(engine, traj, seed_S, catalog):
    step_notes=[]; accepted=[]; final_S=seed_S
    for s in traj:
        if not s.picked and s.status=="exhausted_all": continue
        if s.accepted_id:
            tf=catalog.get(s.accepted_id)
            decision,reason="ACCEPTED",f"best improving move, ΔS>0 with {_rel_of(s)} evidence"
            final_S=round(seed_S + sum(a['delta_S'] for a in accepted) + (s.delta_S or 0),3)
        else:
            tf=catalog.get(s.picked[0]) if s.picked else None
            decision,reason="REJECTED","no candidate passed ΔS>0 + reliability gate (stalled)"
        fact={"current":s.current_smiles,
              "label":(tf.label if tf else (s.picked[0] if s.picked else "?")),
              "family":(tf.family if tf else "?"),
              "move_type":("data-mined" if (tf and tf.family=="discovered") else "standard"),
              "observed_delta_S":(tf.observed_delta_S if tf else None),
              "evidence_smiles":(tf.evidence_smiles if tf else None),
              "rationale":(tf.rationale if tf else None),
              "delta_S":s.delta_S if s.delta_S is not None else "n/a",
              "reliability":_rel_of(s),"tier":"measured",
              "decision":decision,"decision_reason":reason}
        note=engine.complete(SYSTEM_STEP, build_step_prompt(fact))
        step_notes.append({"step":s.step,"action":fact["label"],"decision":decision,"rationale":note.strip()})
        if s.accepted_id:
            accepted.append({"label":fact["label"],"delta_S":s.delta_S,"reliability":_rel_of(s)})
    rej_fams=set()
    for s in traj:
        if s.avoid_after: rej_fams |= set(s.avoid_after)
    stop_reason=traj[-1].status if traj else "n/a"
    summary=engine.complete(SYSTEM_TRAJ, build_traj_prompt(seed_S,final_S,accepted,rej_fams,stop_reason)).strip()
    return {"steps":step_notes,"summary":summary,"seed_S":seed_S,"final_S":final_S}
