from __future__ import annotations
from dataclasses import dataclass
import statistics as st
from rdkit import Chem
from stage_a.chemistry.similarity import tanimoto

@dataclass
class ScoreResult:
    estimated_S: float | None
    delta_S: float | None
    n_neighbors: int
    max_similarity: float
    mean_similarity: float
    neighbor_S_std: float | None
    reliability: str          # exact / high / medium / low / none
    tier: str

def _canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s

class MeasuredNeighborScorer:
    def __init__(self, pool, min_similarity=0.5, k=10):
        recs = pool[["canonical_smiles","selectivity"]].dropna().to_dict("records")
        self.records = []; self.exact = {}
        for r in recs:
            c = _canon(r["canonical_smiles"])
            self.records.append({"canonical_smiles": c, "selectivity": r["selectivity"]})
            self.exact[c] = r["selectivity"]
        self.min_similarity = min_similarity; self.k = k

    def estimate_S(self, smiles):
        c = _canon(smiles)
        # 1) 후보가 pool에 이미 측정돼 있으면 gold 값 직접 사용
        if c in self.exact:
            return self.exact[c], ScoreResult(round(self.exact[c],3),None,1,1.0,1.0,0.0,"exact","measured")
        # 2) 아니면 이웃 추정 (canonical 기준 자기 제외)
        s = [(tanimoto(c, r["canonical_smiles"]), r) for r in self.records if r["canonical_smiles"] != c]
        s = [(a,r) for a,r in s if a >= self.min_similarity]
        s.sort(key=lambda x: x[0], reverse=True); n = s[:self.k]
        if not n:
            return None, ScoreResult(None,None,0,0.0,0.0,None,"none","predictor_only")
        w = [a for a,_ in n]; v = [r["selectivity"] for _,r in n]
        est = sum(a*b for a,b in zip(w,v))/sum(w)
        std = st.pstdev(v) if len(v) > 1 else 0.0
        ms, mn = n[0][0], sum(w)/len(w)
        rel = "high" if (ms>=0.7 and len(n)>=3 and std<0.5) else "medium" if (ms>=0.55 and len(n)>=2) else "low"
        return est, ScoreResult(round(est,3),None,len(n),round(ms,3),round(mn,3),round(std,3),rel,"measured")

    def score(self, cand, seed_smiles=None):
        ec, res = self.estimate_S(cand)
        if seed_smiles and ec is not None:
            es, _ = self.estimate_S(seed_smiles)
            if es is not None: res.delta_S = round(ec - es, 3)
        return res
