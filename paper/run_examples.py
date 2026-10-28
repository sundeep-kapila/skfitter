"""Run every command-line calculation behind the figures and tables of the SKFitter paper.

    python paper/run_examples.py --jobs 16           # everything, 16 calculations at a time
    python paper/run_examples.py --only fits         # the eight final fits of Table 1
    python paper/run_examples.py --only steps        # decay-law / per-shell models, EuMn2Bi2 build-up
    python paper/run_examples.py --only berry        # the Fe and Pt conductivities of Fig. 3
    python paper/run_examples.py --only check        # quick: the published parameters of the eight final models
    python paper/run_examples.py --systems mnte iron # some systems only
    python paper/run_examples.py --summary           # the errors of every fit and step in examples/*/out
    python paper/run_examples.py --update-reference  # copy the final fits in out/ to reference/

A calculation whose output exists is skipped (--force reruns it), so an interrupted run can be started again.
Each calculation runs on --threads cores (default 1); a calculation starts when the fits it starts from are done,
the long EuMn2Bi2 chain first. Fits use --workers Jacobian threads, EuMn2Bi2 fits --workers-eumn2bi2 (default 4).
Logs go to examples/<id>/out/logs/. Then  python paper/make_figures.py  draws Figs. 2-5 and Table 1.

A full rerun on a 16-core machine (about 6 h, most of it the EuMn2Bi2 chain):

    python paper/run_examples.py --force --jobs 12
    python paper/run_examples.py --summary
    python paper/run_examples.py --update-reference
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    import tomli as tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")
SYSTEMS = ["graphene", "mos2", "platinum", "iron", "cerium", "ceria", "mnte", "eumn2bi2"]
LONG_FIRST = ["eumn2bi2", "mos2", "mnte"]            # start the longest chains first (wall time of their fits)


def summary(systems):
    """The errors of every fit and step whose report is in examples/<id>/out (RMSE all / within E_F +- 1 eV)."""
    print(f"{'calculation':28s} {'param.':>6s} {'all':>8s} {'+-1 eV':>8s} {'time (s)':>9s}")
    for ex in systems:
        for path in sorted(glob.glob(os.path.join(EX, ex, "out", f"{ex}*_report.json"))):
            with open(path, encoding="utf-8") as fh:
                r = json.load(fh)
            print(f"{r.get('name', os.path.basename(path)):28s} {r.get('n_params', 0):6d} {r.get('rmse_meV', 0):8.1f} "
                  f"{r.get('near_fermi_1eV_meV', 0):8.1f} {r.get('wall_s', 0):9.0f}")


def update_reference(systems):
    """Copy out/<id>_fit.json and out/<id>_report.json (the final fit of each example) to reference/."""
    for ex in systems:
        src = [os.path.join(EX, ex, "out", f"{ex}{s}") for s in ("_fit.json", "_report.json")]
        if not all(os.path.isfile(f) for f in src):
            print(f"{ex}: no final fit in out/, reference kept")
            continue
        for f in src:
            shutil.copy2(f, os.path.join(EX, ex, "reference", os.path.basename(f)))
        print(f"{ex}: reference/ updated")


def tasks(systems, parts):
    """{name: (example, skfitter arguments, output that marks it done, [names it needs])}"""
    t = {}
    for ex in systems:
        out = lambda f: os.path.join(EX, ex, "out", f)  # noqa: E731
        fit = f"{ex}:fit"
        if "check" in parts:
            t[f"{ex}:check"] = (ex, ["check", "skfitter.toml"], "", [])
        if "fits" in parts:
            t[fit] = (ex, ["fit", "skfitter.toml"], out(f"{ex}_report.json"), [])
            if ex == "mnte":
                t[f"{ex}:spin"] = (ex, ["bands", "skfitter.toml", "--spin"], out("mnte_bands_spin.dat"), [fit])
            if os.path.isfile(os.path.join(EX, ex, "suggest.toml")):       # the orbitals the split model lacks
                t[f"{ex}:suggest"] = (ex, ["character", "suggest.toml", "--suggest"],
                                      os.path.join(EX, ex, "out_suggest", f"{ex}_split_orbital_suggestions.json"), [])
            if os.path.isfile(os.path.join(EX, ex, "character.toml")):     # Fig. 5(a): orbital character at Γ
                t[f"{ex}:character"] = (ex, ["character", "character.toml", "--k", "Γ"],
                                        os.path.join(EX, ex, "out_character", f"{ex}_character_Γ.json"), [])
        if "steps" in parts:
            steps = os.path.join(EX, ex, "steps")
            for fn in sorted(os.listdir(steps)) if os.path.isdir(steps) else []:
                step = fn[:-5]
                with open(os.path.join(steps, fn), "rb") as f:
                    start = (tomllib.load(f).get("fit") or {}).get("start_from", "")
                need = []                                   # start_from ../out/<name>_fit.json: that fit first
                base = os.path.basename(start)
                if base.endswith("_fit.json"):
                    name = base[:-len("_fit.json")]
                    need = [fit] if name == ex else [f"{ex}:{name[len(ex) + 1:]}"]
                t[f"{ex}:{step}"] = (ex, ["fit", f"steps/{fn}"], out(f"{ex}_{step}_report.json"), need)
            if ex == "mnte":
                t[f"{ex}:shell-spin"] = (ex, ["bands", "steps/shell.toml", "--spin"], out("mnte_shell_bands_spin.dat"),
                                         [f"{ex}:shell"])
        if "berry" in parts:
            with open(os.path.join(EX, ex, "skfitter.toml"), "rb") as f:
                props = tomllib.load(f).get("property") or []
            for i, p in enumerate(props, 1):
                tag = {"ahc": f"sk_ahc_{p.get('component', 'z')}",
                       "shc": f"sk_shc_{p.get('spin', 'z')}_{p.get('alpha', 'x')}{p.get('beta', 'y')}"}.get(p["type"])
                stem = f"{ex}_{p['label']}" if p.get("label") else ex
                done = out(f"{stem}_{tag}.dat") if tag else ""
                t[f"{ex}:property{i}"] = (ex, ["berry", "skfitter.toml", "--index", str(i)], done, [fit])
    for name, (_, _, _, need) in t.items():                 # a needed fit outside this run: assume it is done
        t[name] = t[name][:3] + ([n for n in need if n in t],)
    return t


def run(name, ex, args, threads=1):
    logs = os.path.join(EX, ex, "out", "logs")
    os.makedirs(logs, exist_ok=True)
    n = str(threads)
    env = dict(os.environ, OMP_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n, MKL_NUM_THREADS=n,
               PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    t0 = time.time()
    with open(os.path.join(logs, name.replace(":", "_") + ".log"), "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, "-m", "skfitter", *args], cwd=os.path.join(EX, ex), stdout=log,
                           stderr=subprocess.STDOUT, env=env)
    return r.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=["check", "fits", "steps", "berry"], nargs="+")
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="calculations at a time")
    ap.add_argument("--force", action="store_true", help="rerun calculations whose output exists")
    ap.add_argument("--workers", type=int, default=0, help="Jacobian threads of every fit (skfitter fit --workers)")
    ap.add_argument("--workers-eumn2bi2", type=int, default=4, help="Jacobian threads of the EuMn2Bi2 fits")
    ap.add_argument("--summary", action="store_true", help="print the errors of the fits in examples/*/out and stop")
    ap.add_argument("--update-reference", action="store_true",
                    help="copy the final fits in examples/*/out to examples/*/reference and stop")
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS threads per calculation (default 1; more helps the large EuMn2Bi2 models)")
    a = ap.parse_args()
    if a.summary or a.update_reference:
        (summary if a.summary else update_reference)(a.systems)
        return 0
    t = tasks(a.systems, a.only or ["fits", "steps", "berry"])
    done = {n for n, (_, _, out, _) in t.items() if out and os.path.isfile(out) and not a.force}
    failed = set()
    rank = {ex: i for i, ex in enumerate(LONG_FIRST)}
    todo = sorted((n for n in t if n not in done), key=lambda n: rank.get(t[n][0], len(rank)))
    print(f"{len(todo)} calculations to run ({len(done)} already done), {a.jobs} at a time", flush=True)
    running = {}
    with ThreadPoolExecutor(a.jobs) as pool:
        while todo or running:
            for n in list(todo):
                if len(running) >= a.jobs:
                    break
                need = t[n][3]
                if any(m in failed for m in need):
                    todo.remove(n)
                    failed.add(n)
                    print(f"[{time.strftime('%H:%M:%S')}] skipped {n}: needs {', '.join(m for m in need if m in failed)}")
                elif all(m in done for m in need):
                    todo.remove(n)
                    print(f"[{time.strftime('%H:%M:%S')}] start  {n}: skfitter {' '.join(t[n][1])}", flush=True)
                    w = a.workers_eumn2bi2 if t[n][0] == "eumn2bi2" else a.workers
                    args = list(t[n][1]) + (["--workers", str(w)] if w and t[n][1][0] == "fit" else [])
                    running[pool.submit(run, n, t[n][0], args, a.threads)] = n
            if not running:
                break
            fin, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in fin:
                n = running.pop(fut)
                code, sec = fut.result()
                (done if code == 0 else failed).add(n)
                print(f"[{time.strftime('%H:%M:%S')}] {'done ' if code == 0 else 'FAILED'} {n} ({sec:.0f} s)"
                      + ("" if code == 0 else f"  - see examples/{t[n][0]}/out/logs/"), flush=True)
    print(f"finished: {len(done)} done, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
