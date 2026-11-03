from __future__ import annotations

from pathlib import Path

from .architectures import build_model, compile_model
from .config import ARCHITECTURES, SPLITS, STRATEGIES
from .data import make_dataset, samples_for_split, validate_dataset
from .evaluation import evaluate
from .storage import append_metrics, save_config


def verify_setup(config, architectures=None, strategies=None, splits=None) -> dict:
    config = config.resolved()
    report = validate_dataset(config.dataset_dir, tuple(splits or SPLITS))
    return {"data": report, "architectures": architectures or ARCHITECTURES,
            "strategies": strategies or STRATEGIES}


def run_experiments(config, architectures=None, strategies=None, splits=None):
    import tensorflow as tf

    config = config.resolved()
    architectures = tuple(architectures or ARCHITECTURES)
    strategies = tuple(strategies or STRATEGIES)
    splits = tuple(splits or SPLITS)
    tf.keras.utils.set_random_seed(config.seed)
    validate_dataset(config.dataset_dir, splits)
    save_config(config, config.results_dir)
    jobs = []
    for architecture in architectures:
        for strategy in strategies:
            for split in splits:
                job_id = f"{architecture}_{strategy}_{split}"
                out = config.results_dir / job_id
                train_samples = samples_for_split(config.dataset_dir, split, "train")
                val_samples = samples_for_split(config.dataset_dir, split, "val")
                train_ds = make_dataset(train_samples, config.image_size, config.batch_size, True, config.seed)
                val_ds = make_dataset(val_samples, config.image_size, config.batch_size, False, config.seed)
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience,
                                                     restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(out / "best.keras", monitor="val_loss",
                                                       save_best_only=True),
                ]
                out.mkdir(parents=True, exist_ok=True)
                # Fine-tuning follows the required two stages: first learn the
                # classifier head with ImageNet features frozen, then unfreeze
                # the extractor, recompile, and continue at a low learning rate.
                model = build_model(architecture, "head" if strategy == "fine_tune" else strategy,
                                    config.image_size, config.dropout)
                compile_model(model, config.learning_rate)
                history = model.fit(train_ds, validation_data=val_ds,
                                    epochs=config.head_epochs if strategy == "fine_tune"
                                    else config.head_epochs,
                                    callbacks=callbacks, verbose=1)
                if strategy == "fine_tune":
                    base = model.get_layer(architecture)
                    base.trainable = True
                    compile_model(model, config.fine_tune_learning_rate)
                    history_2 = model.fit(train_ds, validation_data=val_ds,
                                          epochs=config.fine_tune_epochs,
                                          callbacks=callbacks, verbose=1)
                    history.history.update({f"fine_tune_{k}": v for k, v in history_2.history.items()})
                labels = [s.label for s in val_samples]
                metrics = evaluate(model, val_ds, labels, out)
                (out / "history.json").write_text(
                    __import__("json").dumps(history.history, indent=2), encoding="utf-8")
                append_metrics(config.results_dir, {
                    "architecture": architecture, "strategy": strategy, "split": split,
                    **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                    "confusion_matrix": metrics["confusion_matrix"],
                })
                jobs.append(job_id)
    return {"jobs": jobs, "failures": []}
