"""Test helpers: the package from src/ (or an installed skfitter), matplotlib off screen, the example data."""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
BASELINES = os.path.join(ROOT, "tests", "baselines")
os.environ.setdefault("MPLBACKEND", "Agg")


def has(*modules) -> bool:
    """True if every module can be imported."""
    return all(importlib.util.find_spec(m) is not None for m in modules)


HAS_SCIENCE = has("scipy", "pymatgen")
EXAMPLES = os.path.join(ROOT, "tests", "data", "examples")


def data_folder(example_id: str) -> str:
    """The folder with the DFT files of a test example: tests/data/examples/<id>/, or the folder of a shipped example
    (examples/<material>/data) that its example.json names under "data"."""
    import json
    folder = os.path.join(EXAMPLES, example_id)
    spec = os.path.join(folder, "example.json")
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            data = json.load(f).get("data")
        if data:
            return os.path.join(ROOT, *data.split("/"))
    return folder


def example_file(example_id: str, name: str) -> str:
    """Path of a file of a test example (tests/data/examples/<id>/), unpacking <name>.gz once into a temporary
    folder when the file is kept gzipped; '' if the example does not have it."""
    import gzip
    import shutil
    import tempfile
    folder = data_folder(example_id)
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        return path
    if not os.path.isfile(path + ".gz"):
        return ""
    out_dir = os.path.join(tempfile.gettempdir(), "skfitter_examples", example_id)
    out = os.path.join(out_dir, name)
    if not os.path.isfile(out) or os.path.getmtime(out) < os.path.getmtime(path + ".gz"):
        os.makedirs(out_dir, exist_ok=True)
        with gzip.open(path + ".gz", "rb") as src, open(out + ".part", "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        os.replace(out + ".part", out)
    for other in os.listdir(folder):
        if not other.endswith(".gz") and not os.path.exists(os.path.join(out_dir, other)):
            shutil.copy(os.path.join(folder, other), out_dir)
    return out


def example_dir(example_id: str) -> str:
    """A folder with every file of a test example unpacked (the example folder itself if nothing is gzipped);
    the example's own folder (absent) if the example is not part of the repository."""
    folder = data_folder(example_id)
    if not os.path.isdir(folder):
        return folder
    gz = [n for n in os.listdir(folder) if n.endswith(".gz")]
    if not gz:
        return folder
    return os.path.dirname(example_file(example_id, gz[0][:-3]))
