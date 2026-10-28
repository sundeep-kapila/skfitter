"""Engines of the property calculators. The native (numpy) engine is the only one in this release; the hooks stay
so that a calculator can list others in ``engines`` and ``P.run(..., engine=...)`` can pick one.
"""
from __future__ import annotations

TITLES = {"native": "Native (numpy)"}


def status(refresh: bool = False) -> dict:
    """{engine: {ok, title, version, message}}."""
    return {"native": {"ok": True, "title": TITLES["native"], "version": None, "message": ""}}


def check(engine: str) -> None:
    """Raise ValueError if ``engine`` is unknown or cannot be used here."""
    if engine not in TITLES:
        raise ValueError(f"Unknown engine {engine!r} (engines: {', '.join(TITLES)}).")


def adjust_cost(engine: str, calc_id: str, cost):
    """The cost estimate of ``calc_id`` run by ``engine``."""
    return cost
