from __future__ import annotations
from dataclasses import dataclass

REL_WEIGHT = {"exact": 1.0, "high": 1.0, "medium": 0.6, "low": 0.3, "none": 0.0}

@dataclass
class Candidate:
    id: str
    product: str
    delta_S: float | None
    reliability: str
    tier: str

@dataclass
class CriticResult:
    accepted: Candidate | None
    score: float | None
    status: str
    should_reflect: bool
    note: str

class DeterministicCritic:
    def __init__(self, min_reliability=("exact","high","medium"), stall_window=3, stall_eps=0.05):
        self.ok = set(min_reliability)
        self.sw = stall_window
        self.se = stall_eps
    def evaluate(self, candidates, history):
        imp = [c for c in candidates
               if c.delta_S is not None and c.delta_S > 0
               and c.reliability in self.ok
               and c.tier == "measured"]
        sc = lambda c: c.delta_S * REL_WEIGHT.get(c.reliability, 0.0)
        imp.sort(key=sc, reverse=True)
        if not imp:
            return CriticResult(None, None, "stalled", True, "개선 후보 없음 → 전략 전환")
        b = imp[0]
        recent = history[-self.sw:]
        stalled = len(recent) >= self.sw and (max(recent)-min(recent) < self.se) and sc(b) < self.se
        return CriticResult(b, round(sc(b),3), "stalled" if stalled else "improved", stalled, f"채택 {b.id}")
