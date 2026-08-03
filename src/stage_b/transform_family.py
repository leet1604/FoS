from __future__ import annotations

from typing import Any


FAMILIES: tuple[str, ...] = (
    "halogen",
    "fluorination",
    "polarity",
    "heteroatom",
    "ring",
    "linker",
    "substituent",
    "retrieval",
    "discovered",
    "unknown",
)


def normalize_family(value: str | None) -> str:
    if not value:
        return "unknown"
    lowered = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fluoro": "fluorination",
        "halogenation": "halogen",
        "polar": "polarity",
        "heterocycle": "heteroatom",
        "scaffold": "ring",
        "side_chain": "substituent",
        "measured_retrieval": "retrieval",
        "dynamic": "discovered",
    }
    family = aliases.get(lowered, lowered)
    return family if family in FAMILIES else "unknown"


def infer_transform_family(rule: dict[str, Any]) -> str:
    explicit = normalize_family(rule.get("family"))
    if explicit != "unknown":
        return explicit

    text = " ".join(
        str(rule.get(key) or "")
        for key in (
            "description",
            "from_frag",
            "to_frag",
            "from_fragment",
            "to_fragment",
            "reaction_smarts",
        )
    ).lower()

    if "fluor" in text or "[f" in text or " f " in f" {text} ":
        return "fluorination"
    if any(token in text for token in ("chlor", "brom", "iod", "halogen", "[cl", "[br", "[i")):
        return "halogen"
    if any(token in text for token in ("hydrox", "methoxy", "amide", "polar", "donor", "acceptor")):
        return "polarity"
    if any(token in text for token in ("pyrid", "aza", "hetero", "n->", "o->", "s->")):
        return "heteroatom"
    if any(token in text for token in ("ring", "morpholine", "piperidine", "piperazine", "cycl")):
        return "ring"
    if any(token in text for token in ("linker", "chain", "spacer")):
        return "linker"
    return "substituent"
