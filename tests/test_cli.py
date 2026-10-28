"""The ``skfitter`` command line on the hBN example: configuration checks, info, character, fit (+ Berry curvature,
Wilson loops and hybrid Wannier centres), bands of the saved fit, and the error messages of a bad configuration."""
import contextlib
import io
import json
import os

import pytest

import _paths

pytestmark = pytest.mark.skipif(not _paths.HAS_SCIENCE, reason="needs scipy and pymatgen")

CONFIG = """
name = "hbn"
output = "out"

[dft]
code = "espresso"
xml = "{d}/qe_bands.xml"
bands_in = "{d}/qe_bands.in"
scf = "{d}/qe_scf.out"
projwfc = "{d}/qe_projwfc_up"

[model]
orbitals = {{ B = ["pz"], N = ["pz"] }}
bands_below = 1
bands_above = 1
neighbours = 3
overlap = true
model_states_only = true

[fit]
starts = 2

[[property]]
type = "berry_path"

[[property]]
type = "wilson"
wnr = 6

[[property]]
type = "wcc"
n_loop = 40
n_scan = 11
"""


def run(*argv):
    from skfitter.cli import main
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(argv))
    return code, out.getvalue()


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    d = _paths.example_dir("hbn-valley").replace("\\", "/")
    p = tmp_path_factory.mktemp("cli") / "hbn.toml"
    p.write_text(CONFIG.format(d=d), encoding="utf-8")
    return str(p)


def test_load_config(config):
    from skfitter.cli import load_config
    c = load_config(config)
    assert c["cfg"]["orbital_config"] == {"B": ["pz"], "N": ["pz"]}
    assert c["cfg"]["non_orthogonal"] is True and c["cfg"]["n_neighbors"] == 3
    assert c["opts"]["n_starts"] == 2
    assert [p["type"] for p in c["properties"]] == ["berry_path", "wilson", "wcc"]
    assert c["properties"][1]["mode"] == "wilson" and c["properties"][1]["wnr"] == 6


def test_info_and_character(config):
    code, out = run("info", config, "--params")
    assert code == 0
    assert "BN, P-6m2" in out and "v_B_pz-N_pz_ppp_0" in out
    code, out = run("character", config, "--k", "K")
    assert code == 0 and "N p" in out
    with open(os.path.join(os.path.dirname(config), "out", "hbn_character_K.json"), encoding="utf-8") as f:
        ch = json.load(f)
    vb, cb = ch["rows"]                                   # valence band on N, conduction band on B at K
    assert vb["weights"]["N p"] > 0.9 and cb["weights"]["B p"] > 0.9


def test_suggest_orbitals(config, tmp_path):
    """character --suggest: the pz model of hBN lacks nothing on average over its window (B s reaches the window at
    a few k-points only); with B given only an s orbital, B p is suggested by both methods."""
    code, out = run("character", config, "--suggest")
    assert code == 0, out
    assert "weighted >= 0.1: nothing to add" in out
    with open(os.path.join(os.path.dirname(config), "out", "hbn_orbital_suggestions.json"), encoding="utf-8") as f:
        r = json.load(f)
    assert {"B p", "N p"} <= set(r["model_channels"]) and r["suggested"]["weighted"] == []
    text = open(config, encoding="utf-8").read().replace('orbitals = {{ B = ["pz"], N = ["pz"] }}'.replace("{{", "{").replace("}}", "}"),
                                                             'orbitals = { B = ["s"], N = ["pz"] }')
    p = tmp_path / "hbn_n.toml"
    p.write_text(text.replace('output = "out"', 'output = "out_n"').replace('name = "hbn"', 'name = "hbn_n"'), encoding="utf-8")
    code, out = run("character", str(p), "--suggest")
    assert code == 0, out
    with open(tmp_path / "out_n" / "hbn_n_orbital_suggestions.json", encoding="utf-8") as f:
        r = json.load(f)
    assert r["suggested"]["absolute"][:1] == ["B p"] and "B p" in r["suggested"]["weighted"]
    bp = next(c for c in r["channels"] if c["channel"] == "B p")
    assert bp["missing_share"] > 0.5 and bp["extra_parameters"] > 0
    assert 'orbitals = { B = ["s", "p"], N = ["pz"] }' in out


def test_run_and_bands(config):
    code, out = run("run", config)
    assert code == 0, out
    out_dir = os.path.join(os.path.dirname(config), "out")
    for name in ("hbn_fit.json", "hbn_fit.txt", "hbn_bands.dat", "hbn_bands.png", "hbn_report.json",
                 "hbn_sk_bc_path_z.dat", "hbn_sk_bc_wilson.dat", "hbn_sk_wcc_k3_0_k2.dat", "hbn_sk_wcc_k3_0_k2.png"):
        assert os.path.isfile(os.path.join(out_dir, name)), name
    with open(os.path.join(out_dir, "hbn_report.json"), encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["rmse_meV"] < 150 and rep["n_params"] == 14
    assert "γ(K)" in out
    assert "Chern number (winding of the summed centres): +0.000" in out or "Chern number (winding of the summed centres): -0.000" in out
    code, out = run("bands", config)
    assert code == 0 and "RMSE" in out


def test_start_from_takes_precedence_over_initial(config):
    """[fit.initial] only starts the parameters the previous fit lacks; start_from values are kept (and inside the
    [fit.bounds] given for them)."""
    from skfitter import cli
    out_dir = os.path.join(os.path.dirname(config), "out")
    with open(os.path.join(out_dir, "hbn_fit.json"), encoding="utf-8") as f:
        prev = json.load(f)["parameters"]
    key = next(k for k in prev if k.startswith("e_"))
    txt = open(config, encoding="utf-8").read().replace(
        "[fit]\nstarts = 2", f'[fit]\nstarts = 2\nstart_from = "out/hbn_fit.json"\n\n[fit.initial]\n{key} = 99.0\n\n'
        f'[fit.bounds]\n{key} = [100.0, 101.0]')
    p = os.path.join(os.path.dirname(config), "hbn_restart.toml")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    conf = cli.load_config(p)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fitter = cli._fitter(conf, lambda *a: None)
        init, bounds = cli._starting_values(conf, fitter, None, lambda *a: None)
    assert init[key] == pytest.approx(prev[key])
    lo, hi = bounds[key]
    assert lo <= prev[key] <= hi and hi == pytest.approx(101.0)
    assert "ignored" in buf.getvalue() and key in buf.getvalue()


def test_bad_configuration(tmp_path, capsys):
    from skfitter.cli import main
    p = tmp_path / "bad.toml"
    p.write_text('[dft]\ncode = "espresso"\nxml = "missing.xml"\n[model]\norbitals = { C = ["pz"] }\n', encoding="utf-8")
    assert main(["info", str(p)]) == 2
    assert "file not found" in capsys.readouterr().err
    p.write_text('[dft]\ncode = "abinit"\n', encoding="utf-8")
    assert main(["info", str(p)]) == 2
    p.write_text('[model]\nnonsense = 1\n', encoding="utf-8")
    assert main(["info", str(p)]) == 2
    assert "required" in capsys.readouterr().err


def test_params_listing():
    code, out = run("params")
    assert code == 0 and "harrison" in out


def test_project_lock(tmp_path):
    """Runs sharing an output folder take turns preparing it; the lock is re-entrant and released."""
    import threading
    import time

    from skfitter import workflow as J
    lock = tmp_path / ".skfitter.lock"
    order = []
    with J.project_lock(str(tmp_path)):
        assert lock.exists()
        with J.project_lock(str(tmp_path)):                      # re-entrant in the same process
            pass
        assert lock.exists()

        def other():
            with J.project_lock(str(tmp_path)):
                order.append("other")
        t = threading.Thread(target=other)
        J._LOCKS_HELD.clear()                                     # the thread acts as a second process
        t.start()
        time.sleep(1.0)
        order.append("first")
        J._LOCKS_HELD.add(str(lock))
    t.join(10)
    assert order == ["first", "other"] and not lock.exists()


def test_check_reproduces_the_reference(tmp_path):
    """skfitter check on the graphene example: the published parameters give the reference errors."""
    import shutil
    from skfitter.cli import main
    d = tmp_path / "graphene"
    shutil.copytree(os.path.join(_paths.ROOT, "examples", "graphene"), d)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["check", str(d / "skfitter.toml")])
    assert code == 0, out.getvalue()
    assert "agrees" in out.getvalue()
    assert (d / "out" / "graphene_bands.dat").is_file()


def test_ctrl_c_requests_a_stop_then_aborts():
    """First Ctrl+C during a fit asks the fit to stop and save; a second one aborts."""
    import pytest as _pt
    from skfitter.cli import _CtrlCStop
    s = _CtrlCStop()
    assert not s()
    s._handler(2, None)
    assert s()
    with _pt.raises(KeyboardInterrupt):
        s._handler(2, None)
