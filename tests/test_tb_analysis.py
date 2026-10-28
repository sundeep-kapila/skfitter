"""prayog_tb.analysis (phase M2c): band-edge requests checked and answered in plain JSON.

* Requests get their defaults, malformed ones are refused; the field-point count drives the plan limits.
* Hexagonal 2D lattices: K / K' found by themselves (60° and 120° cells) and the default path is Γ-M-K-Γ.
* Valleys that are not valleys of the model are left out with a warning instead of failing the sweep.
"""
import json

import numpy as np
import pytest

import _paths  # noqa: F401
from skfitter.model import bands as B
from skfitter.model import textbook
from skfitter.model.analysis import analyse, check_request, points


def test_requests():
    r = check_request({"analysis": "field"})
    assert r["sweep"] == {"B_min": 0.0, "B_max": 10.0, "n": 11, "direction": "out-of-plane"} and points(r) == 11
    assert points(check_request({"analysis": "angle", "angles": {"n": 5}})) == 5
    assert points(check_request({})) == 1 and check_request({})["valleys"] == "auto"
    for bad, word in (({"analysis": "x"}, "analysis"), ({"analysis": "field", "sweep": {"n": 1}}, "sweep"),
                      ({"analysis": "field", "sweep": {"direction": "up"}}, "direction"),
                      ({"analysis": "edges", "B": [1, 2]}, "B"), ({"mesh": 200}, "mesh"),
                      ({"valleys": [1, 2]}, "valleys"), ({"colour": 1}, "unknown")):
        with pytest.raises(ValueError, match=word):
            check_request(bad)


def test_hexagonal_valleys_and_path():
    m = textbook.mos2_three_band()
    v = B.suggest_valleys(m)
    assert np.allclose(v["K"], [2 / 3, 1 / 3, 0]) and np.allclose(v["K'"], [1 / 3, 2 / 3, 0])
    assert [p[0] for p in B.default_path(m)] == ["Γ", "M", "K", "Γ", "K'"]


def test_answers_are_plain_json_and_bad_valleys_are_dropped():
    m = textbook.mos2_three_band()
    r = analyse(m, {"analysis": "edges", "B": [0, 0, 10], "valleys": {"K": [2 / 3, 1 / 3, 0], "P": [0.2, 0.1, 0]}})
    json.dumps(r)
    assert set(r["valleys"]) == {"K"} and any("valley P" in w for w in r["warnings"])
    assert r["two_d"] and r["edges"]["gap"] > 1.0
