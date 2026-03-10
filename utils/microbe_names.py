import re
from typing import List


def rename_microbes(features: List[str]) -> List[str]:
    """
    Convert full taxonomy strings to readable genus/species names.
    Handles both ';' and '_' as separators while preserving '__' taxonomy markers.

    Examples
    --------
    'k__Bacteria;p__Firmicutes;c__Clostridia;o__Clostridiales;f__Lachnospiraceae;g__Roseburia;s__intestinalis'
    → 'Roseburia (g), intestinalis (s)'
    """
    renamed = []
    for feature in features:
        if ';' in feature:
            parts = [p.strip() for p in feature.split(';')]
        else:
            parts = re.split(r'(?<!_)_(?!_)', feature)

        def _get(prefix):
            for p in parts:
                p = p.strip()
                if p.startswith(prefix) and '__' in p:
                    return p.split('__', 1)[1]
            return ''

        Class   = _get('c__')
        order   = _get('o__')
        family  = _get('f__')
        genus   = _get('g__')
        species = _get('s__')
        strain  = _get('t__')

        if strain:
            renamed.append(f"{genus} (g), {species} (s), {strain} (t)")
        elif species:
            renamed.append(f"{genus} (g), {species} (s)")
        elif genus:
            renamed.append(f"{genus} (g)")
        elif family:
            renamed.append(f"{family} (f)")
        elif order:
            renamed.append(f"{order} (o)")
        elif Class:
            renamed.append(f"{Class} (c)")
        else:
            renamed.append(feature)

    return renamed
