from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Ensure local repo imports.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Keep caches in-repo.
os.environ.setdefault("PYHEALTH_CACHE_PATH", os.path.join(REPO_ROOT, ".pyhealth_cache"))

from pyhealth.datasets import MIMIC4EHRDataset, split_by_patient, get_dataloader
from pyhealth.models import TPC
from pyhealth.tasks import RemainingLengthOfStayTPC_MIMIC4
from pyhealth.tasks.length_of_stay_prediction import categorize_los
from pyhealth.trainer import Trainer


def bin_remaining_los_days(y_days: np.ndarray) -> np.ndarray:
    """Bin remaining LoS (days) using PyHealth's 10-bin scheme."""
    flat = y_days.reshape(-1)
    out = np.zeros_like(flat, dtype=int)
    for i, v in enumerate(flat):
        if v == 0:
            # padding
            out[i] = -1
        else:
            out[i] = categorize_los(int(np.floor(v)))
    return out.reshape(y_days.shape)


def masked_kappa_and_accuracy(y_true_days: np.ndarray, y_pred_days: np.ndarray) -> Tuple[float, float]:
    """Compute accuracy and Cohen's kappa on binned labels, masking padding."""
    from sklearn.metrics import cohen_kappa_score

    y_true_bin = bin_remaining_los_days(y_true_days)
    y_pred_bin = bin_remaining_los_days(y_pred_days)
    mask = (y_true_bin != -1)
    yt = y_true_bin[mask]
    yp = y_pred_bin[mask]
    acc = float((yt == yp).mean()) if yt.size else float("nan")
    kappa = float(cohen_kappa_score(yt, yp)) if yt.size else float("nan")
    return kappa, acc


def main():
    # Full dataset root (you indicated: datasets/mimic-iv)
    # Expect a version subfolder like 3.1/ with hosp/ and icu/ inside.
    root = os.path.join(REPO_ROOT, "datasets", "mimic-iv", "3.1")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Expected MIMIC-IV root at {root} (with hosp/ and icu/).")

    # Hard requirements for this pipeline (based on `mimic4_ehr.yaml` + our task).
    required_files = [
        ("hosp", "patients.csv.gz"),
        ("hosp", "admissions.csv.gz"),
        ("icu", "icustays.csv.gz"),
        ("hosp", "labevents.csv.gz"),
        ("hosp", "d_labitems.csv.gz"),
        ("icu", "chartevents.csv.gz"),
    ]
    missing = [os.path.join(root, sub, fn) for sub, fn in required_files if not os.path.exists(os.path.join(root, sub, fn))]
    if missing:
        msg = (
            "Your `datasets/mimic-iv/3.1` folder is missing required CSVs needed for the full run.\n"
            "Missing files:\n- " + "\n- ".join(missing) + "\n\n"
            "Fix: download/extract the corresponding MIMIC-IV modules so that `hosp/` and `icu/` contain these files.\n"
            "Once present, re-run this script."
        )
        raise FileNotFoundError(msg)

    cache_dir = os.path.join(REPO_ROOT, ".pyhealth_dataset_cache_full")

    # NOTE: This is still a *minimal* feature set; for paper-faithful replication,
    # you should replace this with the Table-17 feature list.
    labevent_itemids: List[str] = [
        "50824", "52455", "50983", "52623",
        "50822", "52452", "50971", "52610",
        "50806", "52434", "50902", "52535",
        "50803", "50804",
        "50809", "52027", "50931", "52569",
        "50808", "51624",
        "50960",
        "50868", "52500",
        "52031", "50964", "51701",
        "50970",
    ]

    dataset = MIMIC4EHRDataset(
        root=root,
        tables=["patients", "admissions", "icustays", "labevents", "chartevents"],
        dev=False,
        num_workers=max(1, os.cpu_count() // 4),
        cache_dir=cache_dir,
    )

    task = RemainingLengthOfStayTPC_MIMIC4(
        labevent_itemids=labevent_itemids,
        chartevent_itemids=[],
    )
    sd = dataset.set_task(task)

    train_ds, val_ds, test_ds = split_by_patient(sd, ratios=[0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=8, shuffle=False)

    # Paper-ish hyperparameters for MIMIC-IV (Table 7): temp_channels=11, point_channels=5, layers=8, kernel=5, batch=8.
    model = TPC(
        dataset=sd,
        temporal_channels=11,
        pointwise_channels=5,
        num_layers=8,
        kernel_size=5,
        main_dropout=0.0,
        temporal_dropout=0.05,
        use_batchnorm=True,
        final_hidden=36,
    )

    trainer = Trainer(model, metrics=["mae", "mse"], enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=5,
        monitor="mae",
        monitor_criterion="min",
        optimizer_params={"lr": 0.00221},
    )

    y_true, y_pred, _loss = trainer.inference(test_loader)
    kappa, acc = masked_kappa_and_accuracy(y_true, y_pred)
    print("=== Binned remaining-LoS metrics (masking padding) ===")
    print("Kappa:", kappa)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()

