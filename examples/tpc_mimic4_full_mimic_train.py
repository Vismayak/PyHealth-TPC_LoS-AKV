from __future__ import annotations

"""
Full MIMIC-IV training entrypoint for the PyHealth TPC remaining-LoS pipeline.

Important notes
---------------
- The original Rocheteau codebase trains on *preprocessed hourly tensors* extracted via SQL.
  This repo pipeline reads raw `labevents` / `chartevents` CSVs and builds hourly tensors in-task.
  That is correct logically, but **very expensive** on full MIMIC-IV chartevents.

- The paper's Table 17 lists *human-readable labels* (not itemids). This script resolves labels to
  numeric `itemid`s using `hosp/d_labitems.csv.gz` and `icu/d_items.csv.gz`.

Metrics
-------
This is a regression task. In addition to MAE/MSE/MSLE we report:
  - `bin_accuracy`: exact-match accuracy after bucketing remaining LoS into the same 10 bins used
    by PyHealth's `categorize_los` helper (days-based bucketing).

Example
-------
export PYHEALTH_CACHE_PATH="$PWD/.pyhealth_cache"
python3 examples/tpc_mimic4_full_mimic_train.py \
  --ehr_root "/mnt/c/Users/clean/Desktop/mimiciv-20260407T164707Z-3-001/mimiciv/3.1" \
  --cache_dir "$PWD/.pyhealth_dataset_cache_full" \
  --epochs 10 \
  --batch_size 8 \
  --lr 0.00221 \
  --no_chartevents
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error

# Local repo import
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("PYHEALTH_CACHE_PATH", os.path.join(REPO_ROOT, ".pyhealth_cache"))

from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader, split_by_patient  # noqa: E402
from pyhealth.models import TPC  # noqa: E402
from pyhealth.tasks import RemainingLengthOfStayTPC_MIMIC4  # noqa: E402
from pyhealth.tasks.length_of_stay_prediction import categorize_los  # noqa: E402
from pyhealth.trainer import Trainer  # noqa: E402


TABLE17_CHART_LABELS: List[str] = [
    "Activity / Mobility (JH-HLM)",
    "Mean Airway Pressure",
    "Resp Alarm - High",
    "Apnea Interval",
    "Minute Volume",
    "Resp Alarm - Low",
    "Arterial Blood Pressure Alarm - High",
    "Minute Volume Alarm - High",
    "Respiratory Rate",
    "Arterial Blood Pressure Alarm - Low",
    "Minute Volume Alarm - Low",
    "Respiratory Rate (Set)",
    "Arterial Blood Pressure diastolic",
    "Non Invasive Blood Pressure diastolic",
    "Respiratory Rate (Total)",
    "Arterial Blood Pressure mean",
    "Non Invasive Blood Pressure mean",
    "Respiratory Rate (spontaneous)",
    "Arterial Blood Pressure systolic",
    "Non Invasive Blood Pressure systolic",
    "Richmond-RAS Scale",
    "Braden Score",
    "Non-Invasive Blood Pressure Alarm - High",
    "Strength L Arm",
    "Current Dyspnea Assessment",
    "Non-Invasive Blood Pressure Alarm - Low",
    "Strength L Leg",
    "Daily Weight",
    "O2 Flow",
    "Strength R Arm",
    "Expiratory Ratio",
    "O2 Saturation Pulseoxymetry Alarm - Low",
    "Strength R Leg",
    "Fspn High",
    "O2 saturation pulseoxymetry",
    "Temperature Fahrenheit",
    "GCS - Eye Opening",
    "PEEP set",
    "Tidal Volume (observed)",
    "GCS - Motor Response",
    "PSV Level",
    "Tidal Volume (set)",
    "GCS - Verbal Response",
    "Pain Level",
    "Tidal Volume (spontaneous)",
    "Glucose finger stick (range 70-100)",
    "Pain Level Response",
    "Total PEEP Level",
    "Heart Rate",
    "Paw High",
    "Ventilator Mode",
    "Heart Rate Alarm - Low",
    "Peak Insp. Pressure",
    "Vti High",
    "Heart rate Alarm - High",
    "Phosphorous",
    "Inspired O2 Fraction",
    "Plateau Pressure",
]

TABLE17_LAB_LABELS: List[str] = [
    "Alanine Aminotransferase (ALT)",
    "MCHC",
    "Time in the ICU",
    "Alkaline Phosphatase",
    "MCV",
    "Time of day",
    "Anion Gap",
    "Magnesium",
    "Asparate Aminotransferase (AST)",
    "Oxygen Saturation",
    "Base Excess",
    "PT",
    "Bicarbonate",
    "PTT",
    "Bilirubin, Total",
    "Phosphate",
    "Calcium, Total",
    "Platelet Count",
    "Calculated Total CO2",
    "Potassium",
    "Chloride",
    "Potassium, Whole Blood",
    "Creatinine",
    "RDW",
    "Free Calcium",
    "RDW-SD",
    "Glucose",
    "Red Blood Cells",
    "H",
    "Sodium",
    "Hematocrit",
    "Sodium, Whole Blood",
    "Hematocrit, Calculated",
    "Temperature",
    "Hemoglobin",
    "Urea Nitrogen",
    "I",
    "White Blood Cells",
    "INR(PT)",
    "pCO2",
    "L",
    "pH",
    "Lactate",
    "pO2",
    "MCH",
]


def _norm_label(s: str) -> str:
    return " ".join(s.strip().split()).casefold()


def resolve_itemids_from_dictionary(
    *,
    ehr_root: str,
    chart_labels: Sequence[str],
    lab_labels: Sequence[str],
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Map label strings to itemid strings using MIMIC dictionaries.

    Returns:
      chart_itemids, lab_itemids, ambiguities
    """
    d_items_path = os.path.join(ehr_root, "icu", "d_items.csv.gz")
    d_lab_path = os.path.join(ehr_root, "hosp", "d_labitems.csv.gz")

    # MIMIC dictionary CSVs can contain mixed numeric types in some columns (e.g. highnormalvalue),
    # which can break Polars' initial schema inference if infer_schema_length is too small.
    d_items = pl.read_csv(d_items_path, infer_schema_length=100_000).select(
        pl.col("itemid").cast(pl.Int64).cast(pl.Utf8),
        pl.col("label").cast(pl.Utf8),
    )
    d_lab = pl.read_csv(d_lab_path, infer_schema_length=100_000).select(
        pl.col("itemid").cast(pl.Int64).cast(pl.Utf8),
        pl.col("label").cast(pl.Utf8),
    )

    amb: Dict[str, List[str]] = {}

    def resolve_one(table: pl.DataFrame, label: str) -> str | None:
        lab_n = _norm_label(label)
        hits = table.filter(pl.col("label").map_elements(_norm_label, return_dtype=pl.Utf8) == lab_n)
        if hits.height == 0:
            return None
        if hits.height > 1:
            amb[label] = hits["itemid"].to_list()
            # Deterministic tie-break: smallest itemid
            return str(sorted(hits["itemid"].to_list())[0])
        return str(hits["itemid"][0])

    chart_ids: List[str] = []
    for lab in chart_labels:
        iid = resolve_one(d_items, lab)
        if iid is not None:
            chart_ids.append(iid)

    lab_ids: List[str] = []
    for lab in lab_labels:
        iid = resolve_one(d_lab, lab)
        if iid is not None:
            lab_ids.append(iid)

    # de-dupe while preserving order
    def dedupe(xs: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return dedupe(chart_ids), dedupe(lab_ids), amb


def msle(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.maximum(y_true, eps)
    y_pred = np.maximum(y_pred, eps)
    return float(np.mean((np.log(y_pred) - np.log(y_true)) ** 2))


def bucketize_remaining_los_days(y_days: np.ndarray) -> np.ndarray:
    # Convert fractional days -> whole-day-ish buckets consistent with categorize_los()
    # (This matches the paper's discrete kappa setup at a coarse level.)
    y_hours = y_days * 24.0
    y_whole_days = np.floor(y_hours / 24.0).astype(np.int64)
    return np.vectorize(categorize_los)(y_whole_days).astype(np.int64)


def compute_seq_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    m = mask.astype(bool)
    yt = y_true[m]
    yp = y_pred[m]
    out: Dict[str, float] = {}
    out["mae"] = float(mean_absolute_error(yt, yp))
    out["mse"] = float(mean_squared_error(yt, yp))
    out["msle"] = msle(yt, yp)

    yb = bucketize_remaining_los_days(yt)
    ypb = bucketize_remaining_los_days(yp)
    out["bin_accuracy"] = float(accuracy_score(yb, ypb))
    out["cohen_kappa_bins"] = float(cohen_kappa_score(yb, ypb))
    return out


@dataclass
class Args:
    ehr_root: str
    cache_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dev: bool
    no_chartevents: bool
    monitor: str


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--ehr_root", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default=os.path.join(REPO_ROOT, ".pyhealth_dataset_cache_full"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.00221)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dev", action="store_true", help="Limit to 1000 patients (NOT full cohort).")
    p.add_argument(
        "--no_chartevents",
        action="store_true",
        help="Exclude chartevents table from dataset load (much faster; labs-only time series).",
    )
    p.add_argument("--monitor", type=str, default="mae", choices=["mae", "mse"])
    ns = p.parse_args()
    return Args(
        ehr_root=ns.ehr_root,
        cache_dir=ns.cache_dir,
        epochs=int(ns.epochs),
        batch_size=int(ns.batch_size),
        lr=float(ns.lr),
        weight_decay=float(ns.weight_decay),
        dev=bool(ns.dev),
        no_chartevents=bool(ns.no_chartevents),
        monitor=str(ns.monitor),
    )


def main() -> None:
    args = parse_args()

    chart_itemids: List[str] = []
    lab_itemids: List[str] = []
    amb: Dict[str, List[str]] = {}

    if not args.no_chartevents:
        chart_itemids, lab_itemids, amb = resolve_itemids_from_dictionary(
            ehr_root=args.ehr_root,
            chart_labels=TABLE17_CHART_LABELS,
            lab_labels=TABLE17_LAB_LABELS,
        )
        print(f"Resolved chart itemids: {len(chart_itemids)} / {len(TABLE17_CHART_LABELS)} labels")
        print(f"Resolved lab itemids:   {len(lab_itemids)} / {len(TABLE17_LAB_LABELS)} labels")
        if amb:
            print(f"WARNING: {len(amb)} labels had multiple dictionary matches; smallest itemid chosen.")
    else:
        _, lab_itemids, amb = resolve_itemids_from_dictionary(
            ehr_root=args.ehr_root,
            chart_labels=[],
            lab_labels=TABLE17_LAB_LABELS,
        )
        print("Running labs-only (chartevents excluded).")
        print(f"Resolved lab itemids: {len(lab_itemids)} / {len(TABLE17_LAB_LABELS)} labels")
        if amb:
            print(f"WARNING: {len(amb)} labels had multiple dictionary matches; smallest itemid chosen.")

    tables = ["patients", "admissions", "icustays", "labevents"]
    if not args.no_chartevents:
        tables.append("chartevents")

    ds = MIMIC4EHRDataset(
        root=args.ehr_root,
        tables=tables,
        dev=args.dev,
        num_workers=1,
        cache_dir=args.cache_dir,
    )

    task = RemainingLengthOfStayTPC_MIMIC4(
        labevent_itemids=lab_itemids,
        chartevent_itemids=chart_itemids,
    )

    sample_ds = ds.set_task(task)
    train_ds, val_ds, test_ds = split_by_patient(sample_ds, ratios=[0.7, 0.15, 0.15])

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = TPC(
        dataset=sample_ds,
        temporal_channels=11,
        pointwise_channels=5,
        num_layers=8,
        kernel_size=5,
        main_dropout=0.0,
        temporal_dropout=0.05,
        use_batchnorm=True,
        final_hidden=36,
    )

    trainer = Trainer(model, metrics=["mae", "mse"], enable_logging=True)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=args.epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": args.lr},
        weight_decay=args.weight_decay,
        monitor=args.monitor,
        monitor_criterion="min",
        load_best_model_at_last=True,
    )

    # Extra metrics (MSLE + discrete accuracy) on TEST split using stored best weights.
    y_true_all, y_pred_all, _loss = trainer.inference(test_loader)
    # Build mask from padded labels (padding is 0 in our pipeline)
    mask = y_true_all != 0
    extra = compute_seq_metrics(y_true=y_true_all, y_pred=y_pred_all, mask=mask)

    print("\n=== Additional test metrics (sequence-level, masked) ===")
    for k in sorted(extra.keys()):
        print(f"{k}: {extra[k]:.6f}")


if __name__ == "__main__":
    main()
