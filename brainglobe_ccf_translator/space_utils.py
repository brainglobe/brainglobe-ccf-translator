from __future__ import annotations

import difflib
from typing import Iterable, Set

import pandas as pd

# Map known synonyms to their canonical space names. Keys and values are stored
# in lowercase to make lookups case-insensitive.
SPACE_SYNONYMS = {
    "kim_mouse": "allen_mouse",
    "allen_mouse_bluebrain_barrels": "allen_mouse",
    "osten_mouse": "allen_mouse",
    "demba_allen_seg_dev_mouse":"demba_dev_mouse"
}


def _alias_lookup() -> dict:
    """Return a lower-cased alias mapping."""
    return {k.lower(): v.lower() for k, v in SPACE_SYNONYMS.items()}


def normalise_space_name(space: str) -> str:
    """Normalise a space name using known aliases.

    Parameters
    ----------
    space : str
        User-provided space name.

    Returns
    -------
    str
        Canonicalised space name in lowercase (aliases resolved when present).
    """
    if not isinstance(space, str) or not space.strip():
        raise ValueError("Space name must be a non-empty string.")

    cleaned = space.strip().lower()
    return _alias_lookup().get(cleaned, cleaned)


def normalize_space_name(space: str) -> str:
    """US-spelling wrapper for :func:`normalise_space_name`."""

    return normalise_space_name(space)


def collect_known_spaces(metadata: pd.DataFrame) -> Set[str]:
    """Collect the set of known spaces from translation metadata."""
    if not {"source_space", "target_space"}.issubset(metadata.columns):
        raise ValueError(
            "Metadata must contain 'source_space' and 'target_space' columns."
        )
    source_spaces: Iterable[str] = (
        metadata["source_space"].astype(str).str.lower()
    )
    target_spaces: Iterable[str] = (
        metadata["target_space"].astype(str).str.lower()
    )
    return set(source_spaces) | set(target_spaces)


def collect_known_spaces_with_synonyms(metadata: pd.DataFrame) -> Set[str]:
    """Return known spaces augmented with their accepted synonyms."""

    alias_lookup = _alias_lookup()
    return collect_known_spaces(metadata) | set(alias_lookup.keys())


def validate_space_name(space: str, metadata: pd.DataFrame) -> str:
    """Validate and canonicalise a space name.

    Raises a ValueError if the (normalised) name is unknown.
    """
    canonical_space = normalise_space_name(space)
    alias_lookup = _alias_lookup()
    known_spaces = collect_known_spaces(metadata)
    known_with_synonyms = collect_known_spaces_with_synonyms(metadata)

    if canonical_space in known_spaces:
        return canonical_space

    # If the user typed an alias that normalises to something unknown, fall back to lookup.
    if (
        canonical_space in alias_lookup
        and alias_lookup[canonical_space] in known_spaces
    ):
        return alias_lookup[canonical_space]

    suggestion = difflib.get_close_matches(
        canonical_space, sorted(known_with_synonyms), n=1
    )
    suggestion_msg = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise ValueError(
        "Unknown space '{space}'.{suggestion_msg} Known spaces include: {known}".format(
            space=space,
            known=sorted(known_with_synonyms),
            suggestion_msg=suggestion_msg,
        )
    )
