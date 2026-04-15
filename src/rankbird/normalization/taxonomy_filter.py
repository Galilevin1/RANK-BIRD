"""
Taxonomy-level column filtering for microbiome DataFrames.

Note: kingdom/domain prefix (k__, d__) is stripped at load time in data_loading.py,
so all columns start from phylum level (p__).

Levels
------
None     — keep all columns (no filtering)
"p"      — phylum only: p__ present, c__ absent
"pc"     — phylum + class: p__ present, c__ present, o__ absent
"pco"    — phylum + class + order: p__ present, c__ present, o__ present, f__ absent
"pcof"   — phylum + class + order + family: p__ present, c__ present, o__ present, f__ present, g__ absent
"pcofg"  — phylum + class + order + family + genus: p__ present, c__ present, o__ present, f__ present, g__ present, s__ absent
"g"      — genus only: g__ present, s__ absent
"gs"     — genus + species: g__ present (includes genus-only and genus+species)
"fg"     — family + genus: f__ present, g__ present, s__ absent
"fgs"    — family + genus + species: f__ present, g__ present
"ofg"    — order + family + genus: o__ present, f__ present, g__ present, s__ absent
"cofg"   — class + order + family + genus: c__ present, o__ present, f__ present, g__ present, s__ absent
"""
import re


def has_named(feature: str, prefix: str) -> bool:
    """True if `prefix` is followed by at least one word character."""
    return bool(re.search(re.escape(prefix) + r"\w", feature))


def keep_at_level(feature: str, level) -> bool:
    if level is None:
        return True
    # ── Top-down levels (phylum → genus) ──────────────────────────────────────
    if level == "p":
        return has_named(feature, "p__") and not has_named(feature, "c__")
    if level == "pc":
        return has_named(feature, "p__") and has_named(feature, "c__") and not has_named(feature, "o__")
    if level == "pco":
        return has_named(feature, "p__") and has_named(feature, "c__") and has_named(feature, "o__") and not has_named(feature, "f__")
    if level == "pcof":
        return has_named(feature, "p__") and has_named(feature, "c__") and has_named(feature, "o__") and has_named(feature, "f__") and not has_named(feature, "g__")
    if level == "pcofg":
        return has_named(feature, "p__") and has_named(feature, "c__") and has_named(feature, "o__") and has_named(feature, "f__") and has_named(feature, "g__") and not has_named(feature, "s__")
    # ── Bottom-up levels (genus → family) ─────────────────────────────────────
    if level == "g":
        return has_named(feature, "g__") and not has_named(feature, "s__")
    if level == "gs":
        return has_named(feature, "g__")
    if level == "fg":
        return has_named(feature, "f__") and has_named(feature, "g__") and not has_named(feature, "s__")
    if level == "fgs":
        return has_named(feature, "f__") and has_named(feature, "g__")
    if level == "ofg":
        return has_named(feature, "o__") and has_named(feature, "f__") and has_named(feature, "g__") and not has_named(feature, "s__")
    if level == "cofg":
        return has_named(feature, "c__") and has_named(feature, "o__") and has_named(feature, "f__") and has_named(feature, "g__") and not has_named(feature, "s__")
    return True


def filter_to_level(microbiome_dfs: list, level) -> list:
    """Filter each DataFrame's columns to the requested taxonomy level."""
    if level is None:
        return microbiome_dfs
    filtered = []
    for df in microbiome_dfs:
        cols = [c for c in df.columns if keep_at_level(c, level)]
        filtered.append(df[cols] if cols else df[[]])
    return filtered
