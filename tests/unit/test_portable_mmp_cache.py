import json

import pandas as pd

from stage_a.chemistry.mmp import aggregate_portable_mmp_rules
from stage_a.domain.enums import ConfidenceLabel
from stage_a.domain.models import MMPRule, MMPSupportPair
from stage_a.storage.evidence_cache import (
    PAIR_CACHE_SCHEMA_VERSION,
    EvidenceCacheRepository,
)


def make_exact_rule(
    *,
    rule_id: str,
    core: str,
    from_fragment: str,
    to_fragment: str,
    pair_id: str,
    delta_s: float,
) -> MMPRule:
    pair = MMPSupportPair(
        pair_id=pair_id,
        source_compound=f"{pair_id}:source",
        target_compound=f"{pair_id}:target",
        delta_on=0.1,
        delta_off=0.1 - delta_s,
        delta_selectivity=delta_s,
    )
    return MMPRule(
        rule_id=rule_id,
        core_fragment=core,
        from_fragment=from_fragment,
        to_fragment=to_fragment,
        delta_on=pair.delta_on,
        delta_off=pair.delta_off,
        delta_selectivity=delta_s,
        support_n=1,
        confidence=ConfidenceLabel.LOW,
        supporting_pairs=[pair],
        supporting_pair_ids=[pair_id],
    )


def test_portable_rules_pool_cores_and_share_reverse_family() -> None:
    forward = aggregate_portable_mmp_rules(
        [
            make_exact_rule(
                rule_id="exact:1",
                core="core:1",
                from_fragment="[*:1]C",
                to_fragment="[*:1]N",
                pair_id="pair:1",
                delta_s=0.4,
            ),
            make_exact_rule(
                rule_id="exact:2",
                core="core:2",
                from_fragment="[*:1]C",
                to_fragment="[*:1]N",
                pair_id="pair:2",
                delta_s=-0.3,
            ),
        ],
        minimum_support=2,
    )[0]
    reverse = aggregate_portable_mmp_rules(
        [
            make_exact_rule(
                rule_id="exact:3",
                core="core:1",
                from_fragment="[*:1]N",
                to_fragment="[*:1]C",
                pair_id="pair:3",
                delta_s=-0.4,
            ),
            make_exact_rule(
                rule_id="exact:4",
                core="core:2",
                from_fragment="[*:1]N",
                to_fragment="[*:1]C",
                pair_id="pair:4",
                delta_s=0.3,
            ),
        ],
        minimum_support=2,
    )[0]

    assert forward.support_n == 2
    assert {pair.core_fragment for pair in forward.supporting_pairs} == {
        "core:1",
        "core:2",
    }
    assert forward.transformation_family_id == reverse.transformation_family_id
    assert forward.rule_id != reverse.rule_id


def test_pair_cache_preserves_portable_observation_context(tmp_path) -> None:
    exact_rules = [
        make_exact_rule(
            rule_id=f"exact:{index}",
            core=f"core:{index}",
            from_fragment="[*:1]C",
            to_fragment="[*:1]N",
            pair_id=f"pair:{index}",
            delta_s=delta_s,
        )
        for index, delta_s in enumerate((0.4, -0.3), start=1)
    ]
    portable = aggregate_portable_mmp_rules(
        exact_rules,
        minimum_support=2,
    )
    cache = EvidenceCacheRepository(str(tmp_path))
    saved = cache.save_pair(
        "ON",
        "OFF",
        pd.DataFrame(),
        pd.DataFrame(),
        portable,
        ["fixture"],
    )

    assert set(saved.mmp_pairs["core_fragment"]) == {"core:1", "core:2"}
    assert saved.mmp_pairs["transformation_family_id"].nunique() == 1
    assert cache.has_pair("ON", "OFF")

    manifest_path = cache.pair_paths("ON", "OFF")["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == PAIR_CACHE_SCHEMA_VERSION
