"""Minimal benchmark-side helpers used by :mod:`safecode.eval`.

These replace five imports that previously reached into vendored copies of the
MSSBench and MOSSBench repositories. Only the pieces SafeCoDe actually calls
are reproduced here; see docs/ATTRIBUTION.md for provenance and licensing of
each upstream project.

Deliberately not re-exported at package level -- importing this package must
not trigger network or API-key requirements. Import submodules directly.
"""

__all__: list[str] = []
