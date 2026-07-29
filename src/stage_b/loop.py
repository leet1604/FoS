from __future__ import annotations
from dataclasses import dataclass
from rdkit import Chem
from stage_b.catalog import TransformCatalog
from stage_b.applier import RuleApplier
from stage_b.scorer import MeasuredNeighborScorer
from stage_b.critic import DeterministicCritic, Candidate
from stage_b.discovery import discover_transforms
from stage_b import rule_selector as RS
from stage_b import reflection as RF

def _canon(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else s

@dataclass
class Step:
    step:int; current_smiles:str; picked:list; candidates:list
    accepted_id:str|None; accepted_smiles:str|None; delta_S:float|None
    status:str; n_discovered:int=0; avoid_after:list=None

def run_stage_b(seed, pool, engine, catalog, max_steps=6, n_pick=4,
                rediscover=True, min_gain=0.3):
    records=pool[["canonical_smiles","selectivity"]].dropna().to_dict("records")
    scorer=MeasuredNeighborScorer(pool); applier=RuleApplier(); critic=DeterministicCritic()
    current=_canon(seed); avoid=set(); hist=[]
    all_fam={t.family for t in catalog.transforms}; tried=set(); traj=[]
    visited={current}; seen_tf=set()
    for i in range(max_steps):
        n_new=0
        if rediscover:
            cur_S,_=scorer.estimate_S(current)
            if cur_S is not None:
                for t in discover_transforms(current, records, cur_S, min_gain=min_gain):
                    key=(t.core_fragment,t.from_fragment,t.to_fragment)
                    if key not in seen_tf:
                        t.id=f"disc_s{i}_{len(seen_tf)}"; seen_tf.add(key)
                        catalog.add([t]); all_fam.add(t.family); n_new+=1
        menu=[m for m in catalog.menu_for(current) if m["family"] not in avoid]
        if not menu:
            traj.append(Step(i,current[:30]+"...",[],[],None,None,None,"exhausted_all",n_new,sorted(avoid))); break
        valid={m["id"] for m in menu}
        raw=engine.complete(RS.SYSTEM, RS.build_user_prompt(current,menu,f"avoid={sorted(avoid)}",n_pick))
        picks=RS.parse_picks(raw,valid) or [{"id":menu[0]["id"],"reason":"fb"}]
        picked_fams=set(); cands=[]
        for pk in picks:
            tf=catalog.get(pk["id"]); tried.add(tf.family); picked_fams.add(tf.family)
            ar=applier.apply(current,tf)
            if not ar.applicable: continue
            for prod in ar.products:
                if _canon(prod) in visited: continue
                sr=scorer.score(prod,seed_smiles=current)
                cands.append(Candidate(pk["id"],_canon(prod),sr.delta_S,sr.reliability,sr.tier))
        cres=critic.evaluate(cands,hist)
        step=Step(i,current[:30]+"...",[p["id"] for p in picks],
                  [(c.id,c.delta_S,c.reliability) for c in cands],None,None,None,cres.status,n_new)
        if cres.accepted is not None:
            step.accepted_id=cres.accepted.id; step.accepted_smiles=cres.accepted.product
            step.delta_S=cres.accepted.delta_S; hist.append(cres.accepted.delta_S)
            current=cres.accepted.product; visited.add(current)
        if cres.should_reflect:
            unused=all_fam-tried
            hrows=[{"id":c[0],"family":(catalog.get(c[0]).family if catalog.get(c[0]) else "?"),
                    "delta_S":c[1] or 0.0,"reliability":c[2]} for c in step.candidates]
            rraw=engine.complete(RF.SYSTEM_REFLECT,RF.build_reflect_prompt(hrows,tried,unused,cres.note))
            refl=RF.parse_reflection(rraw,all_fam)
            if refl and refl["avoid_families"]: avoid|=set(refl["avoid_families"])
            avoid|=picked_fams
        step.avoid_after=sorted(avoid); traj.append(step)
    return traj
