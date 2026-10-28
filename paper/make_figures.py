"""Draw Figs. 2-5, Table 1 and the EuMn2Bi2 suggestion table of the SKFitter paper from the command-line outputs (paper/run_examples.py first).
Fig. 1 is a schematic of the method, not computed."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for script in ("table1.py", "table_suggest.py", "fig2.py", "fig3.py", "fig4.py", "fig5.py"):
    print("--", script)
    try:
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    except SystemExit as exc:                     # an output is missing: say which, go on with the next figure
        print(f"   skipped: {exc}")
