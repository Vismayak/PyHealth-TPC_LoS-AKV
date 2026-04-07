from __future__ import annotations

"""
Minimal example: TPC remaining ICU LoS (MIMIC-IV).

This is the paper-style setting:
  - remaining ICU length-of-stay regression
  - hourly predictions starting at hour 5
  - MSLE loss

You must provide MIMIC-IV roots and itemid lists appropriate for your cohort.
"""

import os
import sys

# Put cache inside repo by default (avoids sandbox permission errors).
os.environ.setdefault("PYHEALTH_CACHE_PATH", os.path.join(os.path.dirname(__file__), "..", ".pyhealth_cache"))

# Ensure we import the *local* repo `pyhealth/` rather than any site-packages install.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pyhealth.datasets import MIMIC4EHRDataset, split_by_patient, get_dataloader
from pyhealth.tasks import RemainingLengthOfStayTPC_MIMIC4
from pyhealth.models import TPC
from pyhealth.trainer import Trainer


def main():
    # Adjust these paths for your environment.
    ehr_root = "~/PyHealth-TPC_LoS-AKV/datasets/mimic-iv-demo/2.2"
    cache_dir = os.path.join(_REPO_ROOT, ".pyhealth_dataset_cache")

    # Minimal, lab-only default (you should replace with the paper's full Table 17 feature list).
    # Using the same lab itemids as `InHospitalMortalityMIMIC4` is a reasonable starting point.
    labevent_itemids = [
        "50824", "52455", "50983", "52623",  # Sodium
        "50822", "52452", "50971", "52610",  # Potassium
        "50806", "52434", "50902", "52535",  # Chloride
        "50803", "50804",                    # Bicarbonate
        "50809", "52027", "50931", "52569",  # Glucose
        "50808", "51624",                    # Calcium
        "50960",                             # Magnesium
        "50868", "52500",                    # Anion Gap
        "52031", "50964", "51701",           # Osmolality
        "50970",                             # Phosphate
    ]

    dataset = MIMIC4EHRDataset(
        root=ehr_root,
        tables=["patients", "admissions", "icustays", "labevents", "chartevents"],
        dev=True,
        num_workers=1,
        cache_dir=cache_dir,
    )

    task = RemainingLengthOfStayTPC_MIMIC4(
        labevent_itemids=labevent_itemids,
        chartevent_itemids=[],
    )
    sample_dataset = dataset.set_task(task)

    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, ratios=[0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=8, shuffle=False)

    model = TPC(dataset=sample_dataset)
    trainer = Trainer(model, metrics=["mae", "mse"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=5,
        monitor="mae",
        monitor_criterion="min",
    )


if __name__ == "__main__":
    main()

