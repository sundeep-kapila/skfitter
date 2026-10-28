"""The ``skfitter`` command: fit Slater-Koster tight-binding models to DFT bands and compute Berry-phase properties.

    skfitter info       CONFIG           the DFT data, the crystal and the model the configuration builds
    skfitter character  CONFIG --k K     orbital character of the fitted DFT bands at one k-point (table + plot)
    skfitter check      CONFIG           the published parameters on the DFT bands (no fit): the paper's errors
    skfitter fit        CONFIG           fit the model; parameters, band data, plot and error report
    skfitter bands      CONFIG           DFT and tight-binding bands of a saved fit (data + plot), no refit
    skfitter berry      CONFIG           the [[property]] calculations of the configuration on a saved fit
    skfitter run        CONFIG           fit, then every [[property]]
    skfitter params                      the bundled parameter sets (starting values)

CONFIG is a TOML file (see docs/manual.md). All outputs go to the folder ``output`` of the configuration (relative to
the configuration file), named after ``name``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:                                       # Python 3.11+
    import tomllib
except ModuleNotFoundError:                # pragma: no cover - Python 3.10
    import tomli as tomllib

# --------------------------------------------------------------------------------------------- configuration
# [model] keys of the configuration file -> the SKFitter / workflow names
MODEL_KEYS = {
    "orbitals": "orbital_config", "magnetic_orbitals": "magnetic_orbitals",
    "bands_below": "n_bands_below", "bands_above": "n_bands_above",
    "neighbours": "n_neighbors", "neighbors": "n_neighbors", "hopping": "hopping_mode",
    "soc": "include_soc", "overlap": "non_orthogonal", "electrons": "tb_nelect",
    "fit_eta": "fit_eta", "eta": "fixed_eta", "symprec": "symprec", "model_states_only": "model_states_only",
    "scissor": "scissor", "onsite_mixing": "onsite_mixing", "soc_orbitals": "soc_orbitals",
}
MODEL_DEFAULTS = {"magnetic_orbitals": {}, "n_bands_below": 4, "n_bands_above": 4, "n_neighbors": 3,
                  "hopping_mode": "decay", "include_soc": False, "non_orthogonal": False, "tb_nelect": None,
                  "fit_eta": True, "fixed_eta": 2.5, "symprec": 1e-3, "model_states_only": False}
# [fit] keys -> workflow fit options
FIT_KEYS = {"method": "method", "starts": "n_starts", "seed": "seed", "workers": "workers",
            "tolerance": "tolerance", "overlap_floor": "overlap_floor",
            "maxiter": "maxiter", "popsize": "popsize",
            "k_stride": "k_stride", "refine": "refine", "refine_max_nfev": "refine_max_nfev",
            "fermi_weight": "fermi_weight", "fermi_window": "fermi_window",
            "symmetry_weight": "symmetry_weight", "symmetry_width": "symmetry_width"}
# [dft] keys -> project file roles
DFT_KEYS = {"espresso": {"xml": "qe_xml", "bands_in": "qe_bands_in", "scf": "qe_scf", "projwfc": "qe_projwfc"},
            "vasp": {"vasprun": "vasprun", "kpoints": "kpoints"}}
# defaults of each [[property]] kind
PROPERTY_DEFAULTS = {
    "berry_path": {"kind": "bc", "mode": "path", "component": "z", "dk": 0.01, "nk": 60, "k3": 0.0},
    "berry_map": {"kind": "bc", "mode": "map", "component": "z", "dk": 0.01, "nk": 60, "k3": 0.0},
    "wilson": {"kind": "bc", "mode": "wilson", "component": "z", "k3": 0.0, "wn": 120, "wnr": 30, "wr": 0.0},
    "chern": {"kind": "bc", "mode": "chern", "component": "z", "k3": 0.0, "nk": 60},
    "spin_berry_path": {"kind": "sbc", "spin": "z", "alpha": "x", "beta": "y", "dk": 0.01, "efermi": 0.0,
                        "smearing": 0.0, "units": "au"},
    "ahc": {"kind": "ahc", "component": "z", "nk": [40, 40, 40], "emin": -1.0, "emax": 1.0, "num_efermi": 201,
            "smearing": 0.01, "refine": True, "bands": "all"},
    "shc": {"kind": "shc", "spin": "z", "alpha": "x", "beta": "y", "nk": [40, 40, 40], "emin": -1.0, "emax": 1.0,
            "num_efermi": 201, "smearing": 0.01, "refine": True},
    "wcc": {"kind": "wcc", "fixed": "k3", "fixed_value": 0.0, "loop": "k2", "n_loop": 60, "n_scan": 41, "full": True,
            "n_occupied": 0},
    "shc_temperature": {"kind": "shc_T", "spin": "z", "alpha": "x", "beta": "y", "nk": [30, 30, 30],
                        "temperatures": [0, 100, 200, 300], "efermi": 0.0, "n_shifts": 4, "refine": True},
}


class ConfigError(ValueError):
    pass


def load_config(path: str) -> dict:
    """Read and check a TOML configuration; paths are made absolute (relative to the configuration file)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    here = os.path.dirname(os.path.abspath(path))
    name = str(raw.get("name") or os.path.splitext(os.path.basename(path))[0])
    out_dir = os.path.join(here, str(raw.get("output", "out")))
    dft = dict(raw.get("dft") or {})
    code = dft.pop("code", "espresso")
    if code not in DFT_KEYS:
        raise ConfigError(f"[dft] code must be 'espresso' or 'vasp', not {code!r}")
    sources = {}
    for key, value in dft.items():
        if key not in DFT_KEYS[code]:
            raise ConfigError(f"[dft] unknown key {key!r} for code {code!r} (expected {', '.join(DFT_KEYS[code])})")
        p = os.path.join(here, str(value))
        if not os.path.isfile(p):
            raise ConfigError(f"[dft] {key}: file not found: {p}")
        sources[DFT_KEYS[code][key]] = p
    need = ("qe_xml",) if code == "espresso" else ("vasprun", "kpoints")
    for role in need:
        if role not in sources:
            key = next(k for k, v in DFT_KEYS[code].items() if v == role)
            raise ConfigError(f"[dft] {key} is required")
    model = dict(MODEL_DEFAULTS)
    for key, value in (raw.get("model") or {}).items():
        if key not in MODEL_KEYS:
            raise ConfigError(f"[model] unknown key {key!r} (expected one of {', '.join(sorted(MODEL_KEYS))})")
        model[MODEL_KEYS[key]] = value
    if "orbital_config" not in model:
        raise ConfigError("[model] orbitals is required, e.g. orbitals = { Mo = [\"d\"], S = [\"p\"] }")
    if model["hopping_mode"] not in ("decay", "shell"):
        raise ConfigError("[model] hopping must be 'decay' or 'shell'")
    fit = dict(raw.get("fit") or {})
    opts = {"method": "ls", "n_starts": 8, "refine_max_nfev": 300, "seed": 0}
    for key in list(fit):
        if key in FIT_KEYS:
            opts[FIT_KEYS[key]] = fit.pop(key)
    if float(opts.get("fermi_weight", 1.0)) > 1.0:
        opts["add_weight_at_fermi_bands"] = True
    if float(opts.get("symmetry_weight", 1.0)) > 1.0:
        opts["add_weight_at_symmetry_points"] = True
    start = {"set": fit.pop("start_set", "harrison"), "from": fit.pop("start_from", None),
             "initial": fit.pop("initial", {}), "bounds": fit.pop("bounds", {}), "fix": fit.pop("fix", {})}
    if start["from"]:
        start["from"] = os.path.join(here, str(start["from"]))
    if fit:
        raise ConfigError(f"[fit] unknown key(s): {', '.join(fit)}")
    props = []
    for i, p in enumerate(raw.get("property") or []):
        p = dict(p)
        kind = p.pop("type", None)
        if kind not in PROPERTY_DEFAULTS:
            raise ConfigError(f"[[property]] #{i + 1}: type must be one of {', '.join(PROPERTY_DEFAULTS)}")
        props.append({**PROPERTY_DEFAULTS[kind], **p, "source": "sk", "type": kind})
    return {"path": os.path.abspath(path), "name": name, "output": out_dir, "code": code, "sources": sources,
            "cfg": model, "opts": opts, "start": start, "properties": props, "raw": raw}


# --------------------------------------------------------------------------------------------- helpers
def _quiet(*_a, **_k):
    pass


class _Log:
    def __init__(self, verbose):
        self.verbose = verbose
        self.out = sys.stdout              # the console as it is now: the fitter's output is captured into this log,
                                           # so printing to the (redirected) sys.stdout would call the log again

    def __call__(self, text="", *_a, **_k):
        if self.verbose:
            print(str(text).rstrip(), file=self.out, flush=True)


def _say(text=""):
    print(text, flush=True)


def _project(conf, log):
    """Copy (unpack) the DFT inputs into the output folder and read them: the workflow's project folder."""
    from skfitter import workflow as J
    os.makedirs(conf["output"], exist_ok=True)
    with J.project_lock(conf["output"]):          # several configurations may share one output folder
        J.write_project_file(conf["output"], dft_code=conf["code"])
        return J.prepare_project(conf["output"], conf["code"], conf["sources"], _quiet, log)


def _fitter(conf, log):
    from skfitter import workflow as J
    with J.captured(log):
        return J.build_fitter(conf["cfg"], conf["output"])


def _starting_values(conf, fitter, dft, log):
    """Initial values and bounds: a bundled parameter set (or the defaults), [fit.initial] / [fit.bounds], then a
    previous fit (start_from), whose values take precedence over [fit.initial] (which then only starts the parameters
    the previous fit does not have; its bounds are widened to include them), and last [fit.fix]."""
    from skfitter import params as SP
    from skfitter import workflow as J
    st = conf["start"]
    keys = fitter._get_param_keys()
    sets = SP.bundled_sets()
    if st["set"] in sets:
        sv = J.starting_values(fitter, conf["cfg"], sets[st["set"]], dft)
        log(f"[skfitter] Starting values: parameter set '{st['set']}'")
    elif st["set"] in ("default", "", None):
        sv = {"initial": J.default_initials(fitter, dft), "bounds": {k: J.default_bound(k) for k in keys}}
        log("[skfitter] Starting values: defaults, on-site energies from the DFT band centres")
    else:
        raise ConfigError(f"[fit] start_set {st['set']!r} is not a bundled set ({', '.join(sets)}) or 'default'")
    init = {k: float(v) for k, v in sv["initial"].items()}
    bounds = {k: (float(b[0]), float(b[1])) for k, b in sv["bounds"].items()}
    from_prev = set()                                  # parameters whose start comes from start_from
    if st["from"]:
        sk = J.get_skfitter()
        prev, _, prev_settings = sk.SKFitter.read_param_file(st["from"])
        prev = fitter.convert_legacy_params(prev, prev_settings)     # names before the element-pair integrals
        prev_decay = (prev_settings or {}).get("hopping_mode", "decay") == "decay" if prev_settings else \
            any(k.startswith("eta_") for k in prev)
        if conf["cfg"].get("hopping_mode") == "shell" and prev_decay:
            prev = {**fitter.shell_start_from_decay(prev), **prev}
        prev = {**fitter.split_start(prev), **prev}      # split groups start at their family's values
        n = 0
        for k in keys:
            if k in prev:
                from_prev.add(k)
                v = float(prev[k])
                lo, hi = bounds.get(k, (v - 1.0, v + 1.0))
                if k.startswith(("v_", "s_")):
                    w = max(1.0, 0.6 * abs(v))
                    lo, hi = min(lo, v - w), max(hi, v + w)
                bounds[k], init[k] = (min(lo, v), max(hi, v)), v
                n += 1
        log(f"[skfitter] {n} of {len(keys)} starting values from {os.path.basename(st['from'])}")
    unknown = [k for k in list(st["initial"]) + list(st["bounds"]) + list(st["fix"]) if k not in keys]
    if unknown:
        raise ConfigError("[fit] parameter(s) not in this model: " + ", ".join(sorted(set(unknown)))
                          + "  (skfitter info CONFIG lists the parameters)")
    kept = sorted(k for k in st["initial"] if k in from_prev)
    if kept:
        _say(f"Note: [fit.initial] ignored for {len(kept)} parameter(s) that start from start_from: " + ", ".join(kept))
    for k, v in st["initial"].items():
        if k not in from_prev:
            init[k] = float(v)
    for k, (lo, hi) in st["bounds"].items():
        lo, hi = float(lo), float(hi)
        if k in from_prev:                             # keep the start_from value inside the bounds
            lo, hi = min(lo, init[k]), max(hi, init[k])
        bounds[k] = (lo, hi)
    for k, v in st["fix"].items():                     # fixed: a bound of +-1e-6 around the value
        v = float(v)
        init[k], bounds[k] = v, (v - 1e-6, v + 1e-6)
    return init, bounds


def _band_outputs(conf, fitter):
    """<name>_bands.dat (path coordinate, DFT and TB bands of the fitted window) and <name>_bands.png."""
    import numpy as np
    from matplotlib.figure import Figure

    from skfitter import workflow as J
    try:
        seg = fitter._get_kpath_segments()
    except Exception:  # noqa: BLE001
        seg = []
    ax = J.compute_path_axis(fitter.kpoints, seg)
    x = np.asarray(ax["x"], float)
    dft = np.asarray(fitter.target_dft_bands, float)
    tb = np.asarray(fitter.tb_bands, float)[:, fitter.target_tb_band_indices]
    base = os.path.join(conf["output"], conf["name"])
    head = (f"SKFitter bands of '{conf['name']}' (eV, relative to E_F); k-path ticks: "
            + " ".join(f"{lab}={t:.5f}" for lab, t in zip(list(ax.get("tick_labels") or []),
                                                                     list(ax.get("ticks", []))))
            + "\nk_distance(1/A) " + " ".join(f"dft_{i}" for i in range(dft.shape[1]))
            + " " + " ".join(f"tb_{i}" for i in range(tb.shape[1])))
    np.savetxt(base + "_bands.dat", np.column_stack([x, dft, tb]), fmt="%.6f", header=head, encoding="utf-8")
    fig = Figure(figsize=(6, 5))
    J.draw_fit(fig, fitter)
    fig.savefig(base + "_bands.png", dpi=200)
    return base + "_bands.dat", base + "_bands.png"


def _at_bounds(params, bounds):
    """Parameters that ended within 0.1 % of the width of their bounds from a bound."""
    out = []
    for k, (lo, hi) in (bounds or {}).items():
        v, w = params.get(k), float(hi) - float(lo)
        if v is not None and w > 1e-3 and (v - lo < 1e-3 * w or hi - v < 1e-3 * w):
            out.append(k)
    return out


def _report(conf, fitter, r, wall):
    rep = fitter.fit_error_report(fermi_window=1.0)
    out = {"name": conf["name"], "n_params": len(fitter._get_param_keys()), "rmse_meV": round(1e3 * r["rmse"], 2),
           "errors_meV": {k: round(float(v), 2) for k, v in (r.get("report") or {}).items() if v is not None},
           "near_fermi_1eV_meV": round(float(rep["near_fermi"]), 2), "wall_s": round(wall, 1),
           "parameters": {k: float(v) for k, v in fitter.opt_params.items()},
           "at_bounds": _at_bounds(fitter.opt_params, r.get("bounds"))}
    if r.get("cancelled"):
        out["stopped"] = True                       # Ctrl+C: best parameters so far, not converged
    if fitter.non_orthogonal:
        try:
            out["overlap_min_eig"] = {k: round(v, 4) for k, v in fitter.overlap_conditioning().items()}
        except Exception:  # noqa: BLE001
            out["overlap_min_eig"] = None
    path = os.path.join(conf["output"], f"{conf['name']}_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out, path


def _load_fit(conf, log):
    from skfitter import workflow as J
    _project(conf, log)
    return J.load_fit_job(conf["cfg"], conf["output"], conf["name"], log=log)


# --------------------------------------------------------------------------------------------- commands
def cmd_info(conf, args, log):
    import numpy as np

    from skfitter import workflow as J
    dft = _project(conf, log)
    m = dft["meta"]
    _say(f"DFT       {conf['code']}: {len(dft['kcart'])} k-points on the path "
         f"{'-'.join(dft.get('tick_labels') or [])}, E_F = {m.get('efermi', float('nan')):.4f} eV, "
         f"NELECT = {m.get('nelect')}, SOC {'yes' if m.get('soc') else 'no'}, "
         f"magnetic {'yes' if m.get('magnetic') else ('non-collinear' if J.magnetic_possible(m) else 'no')}, "
         f"projections {'yes' if dft.get('P') else 'no'}")
    f = _fitter(conf, log)
    d = J.describe_fitter(f)["info"]
    st = f.structure
    _say(f"Crystal   {st.composition.reduced_formula}, {d['spacegroup']}, {len(st)} sites, "
         f"a b c = {' '.join(f'{v:.4f}' for v in st.lattice.abc)} Å")
    _say(f"Model     {d['base_dim']} orbitals ({d['total_dim']} with spin), {d['n_params']} parameters, "
         f"{d['n_neighbors']} neighbour shells ({conf['cfg']['hopping_mode']}), {d['n_pairs']} bonds, "
         f"N_e(TB) = {d['tb_nelect']:g}" + (f", moments {d['moments']}" if d.get("moments") else ""))
    for i, (pair, sh) in enumerate(sorted(f._pair_shells().items())):      # shells per element pair (0 = nearest)
        _say(("Shells    " if i == 0 else "          ") + f"{pair}: " + ", ".join(f"{n}: {r:.4f} Å" for n, r in enumerate(sh)))
    _say(f"Fit       {d['n_target_bands']} bands x {d['nk']} k-points")
    if args.params:
        _say("\nParameters (initial, bounds of the start set):")
        init, bounds = _starting_values(conf, f, dft, log)
        for k in f._get_param_keys():
            lo, hi = bounds.get(k, (np.nan, np.nan))
            _say(f"  {k:28s} {init.get(k, np.nan):10.4f}   [{lo:9.4f}, {hi:9.4f}]")
    pts = J.kpath_points(dft)
    if pts:
        _say("k-points  " + ", ".join(f"{p['label']} ({' '.join(f'{c:g}' for c in p['frac'])})" for p in pts))
    return 0


def _parse_k(dft, text):
    from skfitter import workflow as J
    pts = {p["label"]: p for p in J.kpath_points(dft)}
    alias = {"G": "Γ", "Gamma": "Γ", "GAMMA": "Γ"}
    name = alias.get(text, text)
    if name in pts:
        return pts[name]["index"], name, 0.0
    try:
        kf = [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        kf = []
    if len(kf) != 3:
        raise ConfigError(f"--k {text!r}: a label of the path ({', '.join(pts)}) or three fractional coordinates")
    j, dist = J.nearest_kpoint(dft, kf)
    return j, "(" + " ".join(f"{v:g}" for v in kf) + ")", dist


def cmd_character(conf, args, log):
    import numpy as np
    from matplotlib.figure import Figure

    from skfitter import workflow as J
    dft = _project(conf, log)
    if getattr(args, "suggest", False):
        return _suggest_orbitals(conf, args, dft, log)
    j, label, dist = _parse_k(dft, args.k)
    if args.bands == "fit":
        w = J.dft_window(dft, bool(conf["cfg"]["include_soc"]), int(conf["cfg"]["n_bands_below"]),
                         int(conf["cfg"]["n_bands_above"]))
        if w is None:
            raise ConfigError("the number of electrons is not known: give --bands as a range, e.g. 10-20")
        idx = w["indices"]
    else:
        a, _, b = args.bands.partition("-")
        idx = list(range(int(a), int(b or a) + 1))
    ch = J.orbital_character(dft, idx, j, min_share=args.min_share)
    if dist > 1e-3:
        _say(f"Nearest k-point of the DFT path to {label}: index {j}, {dist:.4f} 1/Å away")
    _say(f"Orbital character at {label} (path index {j}), bands {idx[0]}-{idx[-1]}:")
    cols = ch["channels"]
    _say(f"{'band':>5} {'E (eV)':>9} " + " ".join(f"{c:>10s}" for c in cols))
    for r in ch["rows"]:
        _say(f"{r['band']:5d} {r['E']:9.3f} " + " ".join(f"{r['weights'].get(c, 0.0):10.3f}" for c in cols))
    base = os.path.join(conf["output"], f"{conf['name']}_character_{label.strip('()').replace(' ', '_')}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"k_label": label, "distance": dist, **ch}, f, indent=1)
    rows = ch["rows"]
    fig = Figure(figsize=(7.2, 0.3 * len(rows) + 1.6))
    ax = fig.add_subplot(111)
    left = np.zeros(len(rows))
    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    for c_i, c in enumerate(cols):
        v = np.array([100.0 * r["weights"].get(c, 0.0) for r in rows])
        colour = "0.85" if c == "unassigned" else ("0.6" if c == "other" else cmap(c_i % 10))
        ax.barh(range(len(rows)), v, left=left, color=colour, label=c, height=0.8)
        left += v
    ax.set_yticks(range(len(rows)), [f"{r['E']:.2f}" for r in rows])
    ax.set_ylabel("E − E$_F$ (eV)")
    ax.set_xlabel("orbital character (%)")
    ax.set_xlim(0, 100)
    ax.set_title(f"{conf['name']} at {label}")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(base + ".png", dpi=200)
    _say(f"Wrote {os.path.basename(base)}.json, .png")
    return 0


def _toml_orbitals(config: dict) -> str:
    return "orbitals = { " + ", ".join(f"{el} = [" + ", ".join(f'"{o}"' for o in labs) + "]"
                                       for el, labs in config.items()) + " }"


def _suggest_orbitals(conf, args, dft, log):
    """skfitter character CONFIG --suggest: the orbital families the model lacks, ranked over the fitted bands."""
    import copy

    from skfitter import workflow as J
    cfg = conf["cfg"]
    if args.bands == "fit":
        w = J.dft_window(dft, bool(cfg["include_soc"]), int(cfg["n_bands_below"]), int(cfg["n_bands_above"]))
        if w is None:
            raise ConfigError("the number of electrons is not known: give --bands as a range, e.g. 10-20")
        idx = w["indices"]
    else:
        a, _, b = args.bands.partition("-")
        idx = list(range(int(a), int(b or a) + 1))
    opts = conf["opts"]
    fw = float(opts.get("fermi_weight", 1.0)) if opts.get("add_weight_at_fermi_bands", float(opts.get("fermi_weight", 1.0)) > 1) else 1.0
    fwin = float(opts.get("fermi_window", 0.5))
    per_state = 1.0 if (cfg["include_soc"] or dft["E"].get("down") is not None) else 2.0
    with J.captured(log):
        f0 = J.build_fitter(cfg, conf["output"])
    n0 = len(f0._get_param_keys())
    orb = {}                                    # the model's orbitals as built (elements not named get defaults)
    for site, labs in zip(f0.structure, f0.site_orbitals):
        orb.setdefault(site.specie.symbol, list(labs))
    thr = J.MIN_MODEL_WEIGHT if cfg.get("model_states_only") else 0.0
    shells = dft["meta"].get("projected_shells")
    if shells is None and conf["sources"].get("qe_projwfc"):        # a project read before the labels were kept
        try:
            sk = J.get_skfitter()
            sym = [s["symbol"] for s in dft["meta"]["sites"]]
            shells = {}
            for atom, _, lab in sk.read_qe_projwfc_labels(conf["sources"]["qe_projwfc"]):
                if 0 <= atom < len(sym):
                    shells.setdefault(sym[atom], set()).add(lab)
        except (OSError, ValueError):
            shells = None
    r = J.suggest_orbitals(dft, idx, orb, cutoff=args.cutoff, fermi_weight=fw, fermi_window=fwin,
                           electrons_per_state=per_state, min_model_weight=thr, projected_shells=shells)
    sk = J.get_skfitter()
    for ch in r["channels"]:                     # the orbital groups the crystal symmetry allows for each family
        if ch["family"] in ("p", "d", "f"):
            ch["symmetry"] = sk.symmetry_groups(f0.structure, ch["element"], ch["family"], float(cfg.get("symprec", 1e-3)))
    extra = {}
    for ch in r["channels"]:
        if ch["in_model"] or max(ch["absolute"], ch["weighted"]) < args.cutoff:
            continue
        c2 = copy.deepcopy(cfg)
        c2["orbital_config"] = {**orb, ch["element"]: orb.get(ch["element"], []) + [ch["family"]]}
        try:
            with J.captured(log):
                extra[ch["channel"]] = len(J.build_fitter(c2, conf["output"])._get_param_keys()) - n0
        except Exception as exc:  # noqa: BLE001 - report, do not stop the listing
            extra[ch["channel"]] = f"n/a ({exc})"
        ch["extra_parameters"] = extra[ch["channel"]]
    _say(f"Orbital families of {conf['name']} over the fitted bands {r['bands'][0]}-{r['bands'][1]} at all {r['n_k']} "
         f"k-points (weights 1 + {fw - 1:g} exp(-(E/{fwin:g} eV)^2)):")
    _say(f"  in the model: {', '.join(r['model_channels']) or 'none'}; {n0} parameters")
    if thr:
        _say(f"  model_states_only: {r['states_left_out']} of {r['states']} states (less than {thr:g} of their weight on "
             "the model's orbitals) left out, as in the fit")
    _say(f"  weight the model's orbitals cannot carry: {r['missing_average']:.3f} (weighted average per state), of "
         f"which {r['unassigned_average']:.3f} is on no atomic orbital (projection spilling)")
    if r.get("unprojected"):
        _say("  valence shells without a projection: " + ", ".join(f"{el} {', '.join(v)}" for el, v in r["unprojected"].items()))
    _say(f"{'channel':>10} {'model':>6} {'absolute':>9} {'weighted':>9} {'average':>8} {'missing':>8} {'centre':>8} "
         f"{'electrons':>9} {'filled':>6} {'+params':>7}  symmetry groups (site)")
    for ch in sorted(r["channels"], key=lambda c: (c["in_model"], -(c["missing_share"] or 0.0), -c["absolute"])):
        ms = "" if ch["missing_share"] is None else f"{100 * ch['missing_share']:.0f}%"
        ce = "" if ch["centre_eV"] is None else f"{ch['centre_eV']:+.2f}"
        xp = ch.get("extra_parameters", "")
        sy = ch.get("symmetry")
        sg = ""
        if sy:
            sg = " ".join(sy["groups"]) + (f" +{sy['mixing']} on-site" if sy["mixing"] else "") + \
                f" ({', '.join(sy['site_symmetry'])})"
        _say(f"{ch['channel']:>10} {'yes' if ch['in_model'] else '':>6} {ch['absolute']:9.3f} {ch['weighted']:9.3f} "
             f"{ch['average']:8.3f} {ms:>8} {ce:>8} {ch['electrons']:9.2f} {100 * ch['filled']:5.0f}% {str(xp):>7}  {sg}")
    _say("  absolute: largest weight in any state; weighted: largest weighted average over k of any band; average: "
         "weighted average over all states; missing: its share of the weight the model cannot carry; centre: "
         f"projection-weighted energy above {r['energy_floor']:+.1f} eV (eV from E_F; a starting on-site energy); "
         "electrons, filled: occupied weight per cell in that range and its share of the family's capacity (a nearly "
         "full family adds 2(2l+1) electrons per site to [model] electrons, a nearly empty one none); +params: "
         "parameters the model gains; symmetry groups: the finest split into orbital groups the crystal symmetry allows "
         "on all sites of the element (in the structure's axes), '+n on-site': crystal-field / mixing terms the site "
         "symmetry still allows inside the shell ([model] onsite_mixing), and the site point group.")
    for m in ("absolute", "weighted"):
        add = r["suggested"][m]
        if not add:
            _say(f"{m} >= {args.cutoff:g}: nothing to add")
            continue
        new = {el: list(v) for el, v in orb.items()}
        for c in add:
            el, f = c.split()
            new.setdefault(el, []).append(f)
        _say(f"{m} >= {args.cutoff:g}: add {', '.join(add)}")
        _say(f"  {_toml_orbitals(new)}")
    if r.get("unprojected_note"):
        _say("note: " + r["unprojected_note"])
    split, mix = {}, []                          # the model's whole families split as the symmetry allows
    sym = {(ch["element"], ch["family"]): ch.get("symmetry") for ch in r["channels"]}
    for el, labs in orb.items():
        out = []
        for lab in labs:
            sy = sym.get((el, lab)) if lab in ("p", "d", "f") else None
            out += sy["groups"] if sy else [lab]
            if sy and sy["mixing"] and el not in mix:
                mix.append(el)
        split[el] = out
    if split != orb or mix:
        _say("model with its shells split by symmetry:")
        _say(f"  {_toml_orbitals(split)}" + (f"\n  onsite_mixing = {json.dumps(mix)}" if mix else ""))
        r["symmetry_split"] = {"orbitals": split, "onsite_mixing": mix}
    path = os.path.join(conf["output"], f"{conf['name']}_orbital_suggestions.json")
    os.makedirs(conf["output"], exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    _say(f"Wrote {os.path.basename(path)}")
    return 0


class _CtrlCStop:
    """Ctrl+C during a fit: stop at the next band evaluation and save the best parameters found so far (the fit
    files and report are written as usual, marked as stopped). A second Ctrl+C aborts at once without saving."""

    def __init__(self):
        self.requested = False
        self._old = None

    def __call__(self):
        return self.requested

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        try:
            sys.__stdout__.write("\n[skfitter] Ctrl+C: stopping after the current step and saving the best parameters "
                                 "so far (Ctrl+C again aborts without saving) ...\n")
            sys.__stdout__.flush()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        import signal
        import threading
        if threading.current_thread() is threading.main_thread():
            self._old = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *exc):
        import signal
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
        return False


def cmd_fit(conf, args, log):
    if getattr(args, "workers", None):
        conf["opts"]["workers"] = int(args.workers)
    from skfitter import workflow as J
    dft = _project(conf, log)
    f = _fitter(conf, log)
    init, bounds = _starting_values(conf, f, dft, log)
    keys = f._get_param_keys()
    _say(f"Fitting {len(keys)} parameters to {len(f.kpoints)} k-points x {len(f.target_tb_band_indices)} bands "
         f"({conf['opts']['method']}, {conf['opts'].get('n_starts', 8)} starts) ...")
    t0 = time.time()
    last = [time.time()]

    out = sys.stdout                     # the fit captures stdout for its own log: progress goes to the real one

    def metric(*a, **_k):
        if time.time() - last[0] < float(os.environ.get("SKFITTER_PROGRESS_S", "60")):
            return
        last[0] = time.time()
        if len(a) >= 2 and isinstance(a[-1], (int, float)):
            print(f"  ... {(time.time() - t0) / 60:6.1f} min, {int(a[0])} evaluations, "
                  f"best weighted RMSE {a[-1]:.1f} meV",
                  file=out, flush=True)

    with _CtrlCStop() as stop:
        r = J.run_fit_job(conf["cfg"], conf["opts"], init, bounds, conf["output"], conf["name"], _quiet, metric,
                          stop, log=log, dft=dft)
    fitter = r["fitter"]
    if r.get("cancelled"):
        _say("Fit stopped by Ctrl+C: the best parameters found so far are saved (not converged).")
    dat, png = _band_outputs(conf, fitter)
    out, rpath = _report(conf, fitter, r, time.time() - t0)
    _say(f"RMSE {out['rmse_meV']:.1f} meV ({out['near_fermi_1eV_meV']:.1f} meV within 1 eV of E_F), "
         f"{out['n_params']} parameters, {out['wall_s']:.0f} s")
    if out["at_bounds"]:
        _say(f"Warning: {len(out['at_bounds'])} parameter(s) ended at a bound: {', '.join(out['at_bounds'][:8])}"
             + (" ..." if len(out["at_bounds"]) > 8 else "") + " - widen [fit.bounds] if the bound is not physical")
    ov = out.get("overlap_min_eig")
    if ov:
        floor = float(getattr(fitter, "overlap_floor", 0.0) or 0.0)
        _say(f"Overlap S(k): smallest eigenvalue {ov['path']:.3f} on the fitted k-points, {ov['bz']:.3f} on random "
             f"k-points of the zone (overlap_floor {floor:g})"
             + (" - S is nearly singular somewhere; raise overlap_floor" if min(ov.values()) < 0.5 * floor else ""))
    _say("Wrote " + ", ".join(os.path.basename(p) for p in (r["files"]["json"], r["files"]["txt"], dat, png, rpath)))
    return 0


def _spin_band_outputs(conf, fitter, dft):
    """<name>_bands_spin.dat: DFT and tight-binding bands of a collinear magnet resolved by spin (sigma_z), for
    altermagnetic or exchange splittings. The tight-binding states are sorted by the sign of <sigma_z>."""
    import numpy as np

    from skfitter import workflow as J
    if dft["E"].get("down") is None:
        raise ConfigError("--spin needs a spin-polarised (collinear) DFT run with separate spin-up and spin-down bands")
    if fitter.non_orthogonal or fitter.total_dim != 2 * fitter.base_dim:
        raise ConfigError("--spin needs an orthogonal model with spin (magnetic orbitals)")
    kc = np.asarray(fitter.kpoints, float)
    H = fitter._tb_orthogonal_hamiltonians(kc)
    N = fitter.base_dim
    up, dn = [], []
    for h in H:
        w, v = np.linalg.eigh(h)
        sz = np.sum(np.abs(v[:N]) ** 2, axis=0) - np.sum(np.abs(v[N:]) ** 2, axis=0)
        up.append(np.sort(w[sz > 0])[:N])
        dn.append(np.sort(w[sz <= 0])[:N])
    nu, nd = min(map(len, up)), min(map(len, dn))
    tu = np.array([u[:nu] for u in up])
    td = np.array([d[:nd] for d in dn])
    Eu, Ed = np.asarray(dft["E"]["up"], float), np.asarray(dft["E"]["down"], float)
    if len(Eu) != len(kc):
        raise ConfigError("the DFT path and the model's k-points differ (k_stride or a filtered path)")
    try:
        seg = fitter._get_kpath_segments()
    except Exception:  # noqa: BLE001
        seg = []
    ax = J.compute_path_axis(fitter.kpoints, seg)
    x = np.asarray(ax["x"], float)
    path = os.path.join(conf["output"], f"{conf['name']}_bands_spin.dat")
    head = (f"SKFitter spin-resolved bands of '{conf['name']}' (eV, relative to E_F); k-path ticks: "
            + " ".join(f"{lab}={t:.5f}" for lab, t in zip(list(ax.get("tick_labels") or []), list(ax.get("ticks", []))))
            + "\nk_distance(1/A) " + " ".join(f"dft_up_{i}" for i in range(Eu.shape[1]))
            + " " + " ".join(f"dft_dn_{i}" for i in range(Ed.shape[1]))
            + " " + " ".join(f"tb_up_{i}" for i in range(nu)) + " " + " ".join(f"tb_dn_{i}" for i in range(nd)))
    np.savetxt(path, np.column_stack([x, Eu, Ed, tu, td]), fmt="%.6f", header=head, encoding="utf-8")
    return path


def cmd_check(conf, args, log):
    """The published parameters (start_from) on the DFT bands, without fitting: the RMSE of the paper in seconds.
    Writes <name>_bands.dat/.png and <name>_report.json as a fit would; compares with reference/<name>_report.json."""
    from skfitter import workflow as J
    src = conf["start"]["from"]
    if not src:
        raise ConfigError("check needs [fit] start_from: the parameter file to evaluate")
    _project(conf, log)
    f = _fitter(conf, log)
    sk = J.get_skfitter()
    params, _, settings = sk.SKFitter.read_param_file(src)
    params = f.convert_legacy_params(params, settings)
    keys = f._get_param_keys()
    missing = [k for k in keys if k not in params]
    if missing:
        raise ConfigError(f"{os.path.basename(src)} lacks {len(missing)} parameters of this model "
                          f"(e.g. {', '.join(missing[:3])}): it was not fitted with this configuration")
    t0 = time.time()
    f.opt_params = {k: float(params[k]) for k in keys}
    with J.captured(log):
        f._compute_tb_bands(f.opt_params)
    r = {"rmse": J.fit_rmse(f), "report": J.error_report(f)}
    dat, png = _band_outputs(conf, f)
    out, rpath = _report(conf, f, r, time.time() - t0)
    _say(f"RMSE {out['rmse_meV']:.1f} meV ({out['near_fermi_1eV_meV']:.1f} meV within 1 eV of E_F), "
         f"{out['n_params']} parameters, from {os.path.relpath(src, os.path.dirname(conf['path']))}")
    ref = os.path.join(os.path.dirname(src), f"{conf['name']}_report.json")
    code = 0
    if os.path.isfile(ref):
        with open(ref, encoding="utf-8") as fh:
            want = json.load(fh)
        diff = max(abs(out["rmse_meV"] - want["rmse_meV"]), abs(out["near_fermi_1eV_meV"] - want["near_fermi_1eV_meV"]))
        ok = diff <= float(args.tolerance)
        _say(f"Reference {want['rmse_meV']:.1f} / {want['near_fermi_1eV_meV']:.1f} meV: "
             + ("agrees" if ok else "DIFFERS") + f" ({diff:.2f} meV)")
        code = 0 if ok else 1
    _say("Wrote " + ", ".join(os.path.basename(p) for p in (dat, png, rpath)))
    return code


def cmd_bands(conf, args, log):
    r = _load_fit(conf, log)
    dat, png = _band_outputs(conf, r["fitter"])
    files = [dat, png]
    if getattr(args, "spin", False):
        from skfitter import workflow as J
        dft = J.prepare_project(conf["output"], conf["code"], conf["sources"], _quiet, log)
        files.append(_spin_band_outputs(conf, r["fitter"], dft))
    _say(f"RMSE {1e3 * r['rmse']:.1f} meV; wrote {', '.join(os.path.basename(f) for f in files)}")
    return 0


class _KProgress:
    """Progress of the Brillouin-zone sums (conductivities): the uniform mesh, then each level of adaptive refinement
    near band crossings, a line every 5 % of a level (or every 10 minutes) with the elapsed time and the time left
    for that level."""

    def __init__(self, step=0.05, every=600.0):
        self.step, self.every, self.level, self.t0 = step, every, None, time.time()

    def __call__(self, level, done, total):
        now = time.time()
        if level != self.level:
            self.level, self.tl, self.next, self.last = level, now, self.step, now
            what = "uniform k-mesh" if level == 0 else f"refinement level {level} (points near band crossings)"
            self._out(f"  {what}: {total} k-points  ({time.strftime('%H:%M')})")
            return
        frac = done / max(total, 1)
        if frac < self.next and now - self.last < self.every and done < total:
            return
        self.last = now
        while self.next <= frac:
            self.next += self.step
        el = now - self.tl
        left = el * (1.0 - frac) / frac if frac > 0 else float("nan")
        self._out(f"    {100 * frac:5.1f}%  level time {_hms(el)}" + ("" if left != left else f", about {_hms(left)} left")
                  + f", total {_hms(now - self.t0)}  ({time.strftime('%H:%M')})")

    @staticmethod
    def _out(text):
        out = sys.__stdout__ or sys.stdout          # the run's own output is captured into the log; this is not
        out.write(text + "\n")
        out.flush()


class _Progress:
    """Progress of a property run on the console: a line every 5 % (or every 10 minutes), with the elapsed time and
    an estimate of the time left. The estimate covers the uniform k-mesh; adaptive refinement near band crossings
    comes on top and is shown as the last few per cent."""

    def __init__(self, step=0.05, every=600.0):
        self.t0, self.step, self.every = time.time(), step, every
        self.next, self.last = step, self.t0

    def __call__(self, frac, label=""):
        now = time.time()
        if frac < self.next and now - self.last < self.every:
            return
        self.last = now
        while self.next <= frac:
            self.next += self.step
        el = now - self.t0
        left = el * (1.0 - frac) / frac if frac > 0.01 else float("nan")
        out = sys.__stdout__ or sys.stdout          # the run's own output is captured into the log; this is not
        out.write(f"  {100 * frac:5.1f}%  elapsed {_hms(el)}" + ("" if left != left else f", about {_hms(left)} left")
                  + f"  ({time.strftime('%H:%M')})\n")
        out.flush()


def _hms(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _run_properties(conf, fitter, only, log, index=None):
    from skfitter import workflow as J
    props = [p for p in conf["properties"] if not only or p["type"] in only or p.get("label") in only]
    if index:
        n = len(conf["properties"])
        bad = [i for i in index if not 1 <= i <= n]
        if bad:
            raise ConfigError(f"--index {bad[0]}: the configuration has {n} [[property]] entries")
        props = [conf["properties"][i - 1] for i in index]
    if not props:
        _say("No [[property]] entries to run" + (f" matching {', '.join(only)}" if only else ""))
        return 0
    fitted = dict(fitter.opt_params)
    for p in props:
        spec = {k: v for k, v in p.items() if k not in ("type", "label", "parameters", "refine_window")}
        change = {k: float(v) for k, v in (p.get("parameters") or {}).items()}
        unknown = [k for k in change if k not in fitted]
        if unknown:
            raise ConfigError(f"[[property]] parameters not in the model: {', '.join(unknown)}")
        fitter.opt_params = {**fitted, **change}           # e.g. a spin-orbit strength other than the fitted one
        seed = f"{conf['name']}_{p['label']}" if p.get("label") else conf["name"]
        _say(f"{J.property_label(spec)}" + (f" [{p['label']}]" if p.get("label") else "")
             + (" with " + ", ".join(f"{k} = {v:g}" for k, v in change.items()) if change else "") + " ...")
        sk = J.get_skfitter().SKFitter
        sk.k_progress = _KProgress() if p["type"] in ("ahc", "shc", "shc_temperature") else None
        sk.refine_window = float(p.get("refine_window", 0.0) or 0.0)
        try:
            out = J.run_property(fitter, spec, conf["output"], seed, _quiet if sk.k_progress else _Progress(), log,
                                 lambda: False)
        finally:
            sk.k_progress, sk.refine_window = None, 0.0
        for k, v in out["summary"]:
            _say(f"  {k}: {v}")
        files = [os.path.basename(x) for x in (out["png"], out["dat"]) if x and os.path.isfile(x)]
        _say(f"  ({out['elapsed']:.1f} s) wrote {', '.join(files)}")
    fitter.opt_params = fitted
    return 0


def cmd_berry(conf, args, log):
    from skfitter import workflow as J
    if getattr(args, "workers", None):            # threads over k-point blocks of the Brillouin-zone sums
        J.get_skfitter().SKFitter.k_workers = int(args.workers)
        _say(f"[skfitter] {int(args.workers)} threads for the k-point sums (set OMP_NUM_THREADS=1 for best use)")
    r = _load_fit(conf, log)
    return _run_properties(conf, r["fitter"], args.only, log, getattr(args, "index", None))


def cmd_run(conf, args, log):
    cmd_fit(conf, args, log)
    if conf["properties"]:
        return cmd_berry(conf, argparse.Namespace(only=None, index=None), log)
    return 0


def cmd_params(_conf, _args, _log):
    from skfitter import params as SP
    for k, s in SP.bundled_sets().items():
        _say(f"{k:24s} {s.get('name', '')}  ({s.get('licence', s.get('license', ''))})".rstrip())
    return 0


COMMANDS = {"info": cmd_info, "character": cmd_character, "check": cmd_check, "fit": cmd_fit, "bands": cmd_bands,
            "berry": cmd_berry,
            "run": cmd_run, "params": cmd_params}


def build_parser():
    from skfitter import __version__
    ap = argparse.ArgumentParser(prog="skfitter", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Configuration format: docs/manual.md. Examples: examples/*/skfitter.toml.")
    ap.add_argument("--version", action="version", version=f"skfitter {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, help_, config=True):
        p = sub.add_parser(name, help=help_, description=help_)
        if config:
            p.add_argument("config", help="TOML configuration file")
        p.add_argument("-v", "--verbose", action="store_true", help="print the fitter's log")
        return p

    p = add("info", "the DFT data, the crystal and the model of a configuration")
    p.add_argument("--params", action="store_true", help="list the parameters with starting values and bounds")
    p = add("character", "orbital character of the DFT bands at one k-point")
    p.add_argument("--k", default="Γ", help="label of the path (Γ, K, M, ...) or 'kx ky kz' fractional (default Γ)")
    p.add_argument("--bands", default="fit", help="'fit' (the fitted window, default) or a range 'i-j' of DFT bands")
    p.add_argument("--min-share", type=float, default=0.02, help="channels below this share go into 'other'")
    p.add_argument("--suggest", action="store_true",
                   help="rank the orbital families the model lacks over all k-points of the fitted bands (no --k)")
    p.add_argument("--cutoff", type=float, default=0.1,
                   help="with --suggest: weight at which an orbital family is suggested (default 0.1)")
    p = add("check", "the published parameters (start_from) on the DFT bands, no fit: compare with reference/")
    p.add_argument("--tolerance", type=float, default=0.5, help="allowed difference from the reference (meV)")
    p = add("fit", "fit the model to the DFT bands")
    p.add_argument("--workers", type=int, help="threads for the least-squares Jacobian (default 1)")
    p = add("bands", "DFT and tight-binding bands of the saved fit")
    p.add_argument("--spin", action="store_true", help="also the spin-resolved bands of a collinear magnet")
    p = add("berry", "the [[property]] calculations on the saved fit")
    p.add_argument("--only", nargs="+", help="run only these property types (or labels)")
    p.add_argument("--index", nargs="+", type=int, help="run only these [[property]] entries (1 = the first)")
    p.add_argument("--workers", type=int, help="threads over k-point blocks for conductivities (default 1)")
    p = add("run", "fit, then every [[property]]")
    p.add_argument("--workers", type=int, help="threads for the least-squares Jacobian (default 1)")
    add("params", "list the bundled parameter sets", config=False)
    return ap


def _utf8_console():
    """Print Γ, σ, Å, ... on consoles whose code page lacks them (Windows cp1252): UTF-8 there, '?' if even that fails."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv=None) -> int:
    _utf8_console()
    ap = build_parser()
    args = ap.parse_args(argv)
    log = _Log(args.verbose)
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        conf = load_config(args.config) if hasattr(args, "config") else None
        return int(COMMANDS[args.command](conf, args, log) or 0)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"skfitter: error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"skfitter: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
