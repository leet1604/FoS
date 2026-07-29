"""Stage B 데모.

기본은 오프라인 HeuristicLLM 으로 fixture(EGFR/HER2) 위에서 루프를 돈다.
Colab 에서 Qwen3 를 쓰려면 아래 ChatLLM 주석을 해제하면 된다.

실행:
    python scripts/run_stage_b_demo.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from stage_a.providers.fixture import SMILES
from stage_a.wiring import build_fixture_dependencies

from stage_b import HeuristicLLM, StageBConfig, run_stage_b
# from stage_b import ChatLLM  # Colab + Ollama/Qwen3 용


def main() -> None:
    cache = Path("data/cache/contexts_stage_b_demo")
    shutil.rmtree(cache, ignore_errors=True)
    deps = build_fixture_dependencies(str(cache))

    config = StageBConfig(max_iterations=6, beam_k=3)

    # --- LLM 선택 -----------------------------------------------------------
    llm = HeuristicLLM(config)                      # 오프라인 기본값
    # llm = ChatLLM(model="qwen3:8b",               # Colab 무료 T4 + Ollama
    #               base_url="http://localhost:11434/v1",
    #               thinking=False)                  # Assess 때만 thinking=True 권장

    result = run_stage_b(
        seed_smiles=SMILES["CHEMBL_M1"],
        on_target="CHEMBL203",                      # EGFR
        dependencies=deps,
        llm=llm,
        config=config,
        auto_approve_top1=True,
        top_k_off_targets=3,
    )

    print("\n================ TRAJECTORY ================")
    for s in result.trajectory:
        print(f"it{s.iteration} [{s.decision}] conf={s.confidence} "
              f"dOn={s.predicted_delta_on} gain={s.predicted_selectivity_gain}")
        print(f"    plan : {s.plan_rationale}")
        print(f"    judge: {s.rationale}")
        if s.filter_reasons:
            print(f"    filter: {s.filter_reasons}")

    print("\n============ FINAL BEAM (-> Stage C) ============")
    for i, b in enumerate(result.final_beam):
        print(f"#{i} score={b.beam_score:+.3f} gain={b.agg_selectivity_gain:+.2f} "
              f"conf={b.evidence_confidence} terminal={b.terminal}({b.terminal_reason})")
        print(f"    smiles = {b.position.canonical_smiles}")
        print(f"    p_on={b.position.p_activity_on} S={b.position.selectivity_S}")


if __name__ == "__main__":
    main()
