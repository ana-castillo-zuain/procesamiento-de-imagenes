"""mAP, curva precisión-recall y el detector como clasificador.

Ultralytics calcula su propio mAP y torchvision no trae uno, así que acá se
implementa el de COCO para una sola clase: emparejamiento codicioso por score,
cada caja real se usa una vez, AP interpolado en 101 puntos de recall y
promedio sobre IoU 0.50, 0.55, ..., 0.95. Sin rangos de área ni ``iscrowd``.
Hasta 100 detecciones por imagen, que es el default de torchvision.

Solo usa numpy: se puede recalcular desde las predicciones guardadas sin GPU.
"""

from __future__ import annotations

import numpy as np


IOU_THRESHOLDS = np.round(np.linspace(0.5, 0.95, 10), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
_EPS = 1e-12


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def _match_image(boxes: np.ndarray, scores: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Verdaderos positivos ``(n_pred, n_iou)`` con las predicciones ordenadas por score.

    Como en COCO: cada predicción, de mayor a menor score, toma la caja real
    libre con más IoU, si ese IoU llega al umbral.
    """
    order = np.argsort(-scores, kind="mergesort")
    boxes, scores = boxes[order], scores[order]
    tp = np.zeros((len(boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(boxes) and len(gt):
        iou = box_iou(boxes, gt)
        for t, threshold in enumerate(IOU_THRESHOLDS):
            taken = np.zeros(len(gt), dtype=bool)
            for d in range(len(boxes)):
                candidates = np.flatnonzero(~taken & (iou[d] >= threshold))
                if candidates.size:
                    taken[candidates[np.argmax(iou[d, candidates])]] = True
                    tp[d, t] = True
    return scores, tp


def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, RECALL_POINTS, side="left")
    valid = index < len(envelope)
    return float(np.where(valid, envelope[np.minimum(index, len(envelope) - 1)], 0.0).mean())


def detection_metrics(predictions: list, ground_truths: list) -> dict:
    """mAP50, mAP50-95 y precisión/recall en el punto de F1 máximo (IoU 0.5).

    ``predictions`` es una lista por imagen de ``(boxes, scores)`` y
    ``ground_truths`` una lista por imagen de ``boxes``, en píxeles ``xyxy``.
    Precisión y recall se toman donde F1 es máximo, como hace Ultralytics, y
    se devuelve la confianza de ese punto.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Hay distinta cantidad de imágenes en predicciones y etiquetas.")
    n_gt = sum(len(gt) for gt in ground_truths)
    if n_gt == 0:
        raise ValueError("No hay cajas reales: el AP no está definido. kingston2 se mide con falsos positivos.")

    scores_list, tp_list = [], []
    for (boxes, scores), gt in zip(predictions, ground_truths):
        scores, tp = _match_image(np.asarray(boxes, dtype=np.float64), np.asarray(scores, dtype=np.float64), np.asarray(gt))
        scores_list.append(scores)
        tp_list.append(tp)
    scores = np.concatenate(scores_list) if scores_list else np.zeros(0)
    tp = np.concatenate(tp_list) if tp_list else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
    order = np.argsort(-scores, kind="mergesort")
    scores, tp = scores[order], tp[order]

    cum_tp = np.cumsum(tp, axis=0)
    cum_fp = np.cumsum(~tp, axis=0)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, _EPS)
    aps = np.array([_ap_101(recall[:, t], precision[:, t]) for t in range(len(IOU_THRESHOLDS))])

    r50, p50 = recall[:, 0], precision[:, 0]
    if len(scores):
        f1 = 2 * p50 * r50 / np.maximum(p50 + r50, _EPS)
        best = int(np.argmax(f1))
        at_best = {"precision": float(p50[best]), "recall": float(r50[best]), "f1": float(f1[best]), "conf_at_best_f1": float(scores[best])}
    else:
        at_best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "conf_at_best_f1": float("nan")}
    return {
        "map50": float(aps[0]),
        "map50_95": float(aps.mean()),
        **at_best,
        "n_gt_boxes": int(n_gt),
        "n_predictions": int(len(scores)),
        "curve": {"recall": r50, "precision": p50, "confidence": scores},
    }


def image_scores(predictions: list) -> np.ndarray:
    """Confianza máxima por imagen, 0 si no hay detecciones.

    Es el score continuo del detector usado como clasificador.
    """
    return np.array([float(np.max(scores)) if len(scores) else 0.0 for _, scores in predictions])


def classification_metrics(scores: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    """Una imagen "tiene paneles" si su confianza máxima llega al umbral."""
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(scores) >= threshold
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))

    def f1_of(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
        precision = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        recall = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision, recall, f1 = f1_of(tp, fp, fn)
    _, _, f1_negative = f1_of(tn, fn, fp)
    return {
        "threshold": float(threshold),
        "n_images": int(len(truth)),
        "n_positive_images": int(truth.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(truth) if len(truth) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Promedio de F1 de las dos clases. Con 89 % de positivas, la accuracy
        # de "todo positivo" ya es alta; el F1 macro no se deja engañar.
        "f1_macro": (f1 + f1_negative) / 2,
    }


def best_threshold(scores: np.ndarray, truth: np.ndarray) -> float:
    """Umbral que maximiza el F1 macro sobre estas mismas imágenes.

    Se elige mirando la validación, así que es un diagnóstico: separa "no
    discrimina" de "discrimina pero el umbral está mal puesto". No es una
    estimación honesta de generalización.
    """
    candidates = np.unique(np.concatenate([[0.0], np.asarray(scores, dtype=np.float64)]))
    values = [classification_metrics(scores, truth, threshold)["f1_macro"] for threshold in candidates]
    return float(candidates[int(np.argmax(values))])
