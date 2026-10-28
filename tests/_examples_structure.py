"""Helpers for the structure / neighbour tests: the DFT data (meta) and the model of each illustrative project."""
import json
import os
import tempfile
from functools import lru_cache

import _paths

QUIET = lambda *a, **k: None  # noqa: E731


def example_ids():
    out = []
    for d in sorted(os.listdir(_paths.EXAMPLES)):
        spec = os.path.join(_paths.EXAMPLES, d, "example.json")
        if os.path.isfile(spec):
            out.append((json.load(open(spec, encoding="utf-8")).get("order", 99), d))
    return [d for _, d in sorted(out)]


def example_spec(example_id):
    path = os.path.join(_paths.EXAMPLES, example_id, "example.json")
    if not os.path.isfile(path):
        import pytest
        pytest.skip(f"example {example_id} is not part of the repository")
    with open(os.path.join(_paths.EXAMPLES, example_id, "example.json"), encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def project(example_id):
    """(project_dir, dft) of an illustrative project: its DFT files read into a temporary project folder."""
    from skfitter import workflow as J
    ex = example_spec(example_id)
    sources = {}
    for role, name in ex["files"].items():
        stem = name[:-3] if name.endswith(".gz") else name
        sources[role] = _paths.example_file(example_id, stem)
    project_dir = tempfile.mkdtemp(prefix=f"prayog-struct-{example_id}-")
    dft = J.prepare_project(project_dir, ex["code"], sources, QUIET, QUIET)
    return project_dir, dft


def cfg(example_id):
    from skfitter import workflow as J
    c = dict(example_spec(example_id)["cfg"])
    c.setdefault("symprec", 1e-3)
    return c


def fitter(example_id, cfg_=None):
    from skfitter import workflow as J
    project_dir, _ = project(example_id)
    with J.captured(QUIET):
        return J.build_fitter(cfg_ or cfg(example_id), project_dir)
