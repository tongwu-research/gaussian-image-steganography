"""Command-line entry points for fitting, embedding and extraction."""
import argparse
from pathlib import Path
import runpy
import sys

COMMANDS = {
    "fit": "run_fit.py",
    "embed": "watermark_experiment_v3.py",
    "extract": "extract.py",
    "demo": "extraction_demo.py",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    # Historical modules use flat imports. Keep the adapter at the CLI boundary.
    sys.path.insert(0, str(source))
    script = source / COMMANDS[args.command]
    sys.argv = [str(script), *args.arguments]
    runpy.run_path(str(script), run_name="__main__")
