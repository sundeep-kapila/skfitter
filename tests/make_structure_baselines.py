"""Write tests/baselines/structure/<example>.json: the cell, shells and per-site neighbour counts of every
illustrative project, as skfitter_structure computes them (checked by test_skfitter_structure.py).

    python tests/make_structure_baselines.py

Review the diff before committing: these files are the reference the structure / neighbour views are held to.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402,F401
from _examples_structure import cfg, example_ids, project  # noqa: E402

OUT = os.path.join(_paths.BASELINES, "structure")


def summary(example_id: str) -> dict:
    from skfitter import structure as S
    _, dft = project(example_id)
    meta, c = dft["meta"], cfg(example_id)
    sv = S.structure_view(meta, float(c.get("symprec", 1e-3)))
    nv = S.neighbour_view(meta, c)
    return {
        "about": "skfitter_structure reference, see tests/make_structure_baselines.py",
        "formula": sv["formula"], "spacegroup": sv["spacegroup"], "n_sites": sv["n_sites"],
        "n_inequivalent": sv["n_inequivalent"], "dimensionality": sv["dimensionality"],
        "abc": sv["abc"], "angles": sv["angles"], "volume": sv["volume"],
        "n_neighbors": nv["n_neighbors"], "n_shells": nv["n_shells"], "cutoff": nv["cutoff"], "d0": nv["d0"],
        "shells": nv["shells"], "pair_shells": {k: [round(x, 4) for x in v] for k, v in nv["pair_shells"].items()},
        "n_pairs": nv["n_pairs"], "n_bonds": len(nv["bonds"]),
        "hopping_keys": [k["key"] for k in nv["hopping_keys"]],
        "sites": [{"label": s["label"], "eq": s["eq"],
                   "shells": [[r["shell"], r["dist"], r["count"], r["by_element"]] for r in s["shells"]]}
                  for s in nv["sites"]],
        "warnings": len(nv["warnings"]),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    for ex in example_ids():
        d = summary(ex)
        with open(os.path.join(OUT, f"{ex}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"{ex:20s} {d['formula']:10s} {d['spacegroup']['symbol']:10s} {d['n_sites']:3d} atoms, "
              f"{d['n_shells']} shells, {d['n_pairs']} pairs")


if __name__ == "__main__":
    main()
