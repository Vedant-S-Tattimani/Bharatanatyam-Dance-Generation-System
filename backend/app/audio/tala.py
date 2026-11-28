"""Tala definitions and beat-grid mapping."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TalaSpec:
    name: str
    beats: int
    angams: tuple[int, ...]
    description: str


TALAS: dict[str, TalaSpec] = {
    "adi": TalaSpec(
        name="Adi",
        beats=8,
        angams=(4, 2, 2),
        description="Laghu (4) + Drutam (2) + Drutam (2) = 8 beats",
    ),
    "rupaka": TalaSpec(
        name="Rupaka",
        beats=6,
        angams=(2, 4),
        description="Drutam (2) + Laghu (4) = 6 beats",
    ),
    "misra_chapu": TalaSpec(
        name="Misra Chapu",
        beats=7,
        angams=(3, 2, 2),
        description="3 + 2 + 2 = 7 beats",
    ),
    "khanda_chapu": TalaSpec(
        name="Khanda Chapu",
        beats=5,
        angams=(2, 3),
        description="2 + 3 = 5 beats",
    ),
    "ata": TalaSpec(
        name="Ata",
        beats=14,
        angams=(5, 5, 2, 2),
        description="Laghu (5) + Laghu (5) + Drutam (2) + Drutam (2) = 14 beats",
    ),
}


def get_tala(name: str = "adi") -> TalaSpec:
    """Return TalaSpec by case-insensitive key, defaulting to Adi tala (8 beats)."""
    key = name.strip().lower().replace(" ", "_").replace("_tala", "")
    if key in TALAS:
        return TALAS[key]
    raise ValueError(f"Unsupported tala '{name}'. Available: {sorted(TALAS.keys())}")

