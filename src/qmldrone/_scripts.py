import argparse
import os
from qmldrone.models.hybrid import create as create_hybrid
from qmldrone.models.classic import create as create_classic

parser = argparse.ArgumentParser(
    description="Build a drone detection or classification model from YAML config"
)

parser.add_argument(
    "type", choices=["hybrid", "classic"], help="Select a classic or hybrid model."
)  # hybrid or classic
parser.add_argument("load", help="File path for YAML config.")  # yaml path


def main(argv: list[str] | None = None) -> int:
    args = parser.parse_args(argv)
    config_path = args.load
    if not os.path.exists(config_path):
        raise FileNotFoundError("YAML config not found")
    if args.type == "classic":
        create_classic(config_path)
    elif args.type == "hybrid":
        create_hybrid(config_path)