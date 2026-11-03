"""Lanza la verificación o los entrenamientos de Faster R-CNN.

Sin ``--train`` no se carga ningún modelo: solo se revisan los datos, la tabla
de congelamiento y la grilla. El notebook de Colab llama a ``run_experiments``
directamente y no necesita editar este archivo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solar_rcnn.architectures import ARCHITECTURES, STRATEGIES
from solar_rcnn.config import Config, default_dataset_dir, default_results_dir
from solar_rcnn.engine import run_experiments, verify_setup


def main() -> None:
    parser = argparse.ArgumentParser(description="Detección de paneles solares con Faster R-CNN, por fuente.")
    parser.add_argument("--dataset", type=Path, default=None, help="Carpeta con PV01_val, solarpanels_val y geo_kingston_val.")
    parser.add_argument("--results", type=Path, default=None, help="Carpeta results/. En Colab tiene que estar en Drive.")
    parser.add_argument("--architectures", nargs="*", default=None, choices=list(ARCHITECTURES))
    parser.add_argument("--strategies", nargs="*", default=None, choices=list(STRATEGIES))
    parser.add_argument("--folds", nargs="*", type=int, default=None, help="1, 2 y 3.")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Entrena. Sin esta bandera el script solo verifica datos y tablas, sin cargar modelos.",
    )
    parser.add_argument("--rerun", action="store_true", help="No saltea los experimentos ya terminados.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    config = Config(
        dataset_dir=args.dataset or default_dataset_dir(),
        results_dir=args.results or default_results_dir(),
        resume=not args.rerun,
        allow_cpu=args.allow_cpu,
    )
    if args.batch is not None:
        config.batch = args.batch
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.workers is not None:
        config.workers = args.workers

    if not args.train:
        print(verify_setup(config, args.architectures, args.strategies, args.folds))
        print("\nVerificación estática terminada. No se cargó ningún modelo. Para entrenar, agregá --train.")
        return

    outcome = run_experiments(
        config,
        architectures=args.architectures,
        strategies=args.strategies,
        folds=args.folds,
        stop_on_error=args.stop_on_error,
    )
    print(f"Experimentos en la grilla: {len(outcome['jobs'])}")
    if outcome["failures"]:
        print(f"Fallaron {len(outcome['failures'])}:")
        for failure in outcome["failures"]:
            print(f"  {failure['experiment_id']}: {failure['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
