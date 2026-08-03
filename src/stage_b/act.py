from __future__ import annotations

from .config import StageBConfig
from .observe import Observation
from .schemas import CandidateEdit, Position

# RDKit / SA score 는 선택적 (없어도 루프는 돈다).
try:  # pragma: no cover
    from rdkit import Chem
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False

try:  # pragma: no cover
    from rdkit.Chem import RDConfig
    import os
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
    _HAS_SASCORE = True
except Exception:  # pragma: no cover
    _HAS_SASCORE = False


def generate(edit: CandidateEdit) -> str:
    """Stage A 가 이미 만든 generated_product 를 채택 (재생성 없음)."""
    return edit.product_smiles


def predict_position(obs: Observation, edit: CandidateEdit) -> Position:
    """MMP 통계 기반 예측 위치.

    new_p_on  = p_on + delta_on
    new_p_off = p_off + delta_off  (off 별)
    new_S     = new_p_on - new_p_off
    (향후 ligand/structure predictor 훅 위치)
    """
    p_on = obs.p_activity_on
    new_p_on = (p_on + edit.delta_on) if p_on is not None else None

    new_p_off: dict[str, float | None] = {}
    new_S: dict[str, float | None] = {}
    per_off_map = {po.off_target_id: po for po in edit.per_off}
    for off_id, cur_off in obs.p_activity_off.items():
        po = per_off_map.get(off_id)
        if cur_off is not None and po is not None:
            no = cur_off + po.delta_off
        else:
            no = cur_off
        new_p_off[off_id] = no
        if new_p_on is not None and no is not None:
            new_S[off_id] = new_p_on - no
        else:
            new_S[off_id] = None

    return Position(
        canonical_smiles=edit.product_smiles,
        p_activity_on=new_p_on,
        p_activity_off=new_p_off,
        selectivity_S=new_S,
        predicted=True,
    )


def synthesizability(smiles: str) -> tuple[bool, float | None]:
    """(valid, sa_score). sa_score 는 1(쉬움)~10(어려움). RDKit 없으면 (True, None)."""
    if not _HAS_RDKIT:
        return True, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, None
    sa = None
    if _HAS_SASCORE:
        try:
            sa = float(sascorer.calculateScore(mol))
        except Exception:
            sa = None
    return True, sa


def passes_filter(
    obs: Observation,
    edit: CandidateEdit,
    predicted: Position,
    config: StageBConfig,
) -> tuple[bool, list[str]]:
    """하드 제약. 통과 못하면 Plan 이 다음 후보로 RETRY.

    독성/합성가능성 체크는 (설계상) Stage B 에 위치한다.
    """
    reasons: list[str] = []

    # 1) on-target 하락 한도
    if edit.delta_on < -config.max_on_target_drop:
        reasons.append(
            f"on-target drop {edit.delta_on:+.2f} exceeds "
            f"limit -{config.max_on_target_drop}"
        )

    # 2) required off 악화 가드 (선택도 목표와 정면 충돌)
    req_ids = {o.off_id for o in obs.offs if o.requirement in {"required", "selected"}}
    for po in edit.per_off:
        if po.off_target_id in req_ids and po.delta_off > config.max_required_off_worsen:
            reasons.append(
                f"required off {po.off_target_id} worsens "
                f"(dOff={po.delta_off:+.2f} > {config.max_required_off_worsen})"
            )

    # 3) 선택도 개선 최소폭
    if edit.agg_selectivity_gain < config.min_selectivity_gain:
        reasons.append(
            f"selectivity gain {edit.agg_selectivity_gain:+.2f} "
            f"below min {config.min_selectivity_gain}"
        )

    # 4) 합성가능성 / valence sanity
    valid, sa = synthesizability(edit.product_smiles)
    if not valid:
        reasons.append("product fails RDKit sanitization")
    if sa is not None and sa > 6.0:
        reasons.append(f"SA score {sa:.1f} high (hard to synthesize)")

    return (len(reasons) == 0), reasons
