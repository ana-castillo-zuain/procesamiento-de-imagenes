from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solar_classification.config import (ARCHITECTURES, SPLITS, STRATEGIES,
                                         Config, default_dataset_dir,
                                         default_results_dir)
from solar_classification.engine import run_experiments, verify_setup


def main():
    parser = argparse.ArgumentParser(description="Clasificación de paneles con Keras.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_dir())
    parser.add_argument("--results", type=Path, default=default_results_dir())
    parser.add_argument("--architectures", nargs="*", choices=ARCHITECTURES)
    parser.add_argument("--strategies", nargs="*", choices=STRATEGIES)
    parser.add_argument("--splits", nargs="*", choices=SPLITS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    config = Config(args.dataset, args.results, batch_size=args.batch_size)
    if args.epochs is not None:
        config.head_epochs = args.epochs
        config.fine_tune_epochs = args.epochs
    if not args.train:
        print(verify_setup(config, args.architectures, args.strategies, args.splits))
        return
    outcome = run_experiments(config, args.architectures, args.strategies, args.splits)
    print(f"Experimentos completados: {len(outcome['jobs'])}")


if __name__ == "__main__":
    main()
