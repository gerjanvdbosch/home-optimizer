from __future__ import annotations

from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow, MpcDatasetSummary


class IdentificationDatasetRow(MpcDatasetRow):
    pass


class IdentificationDataset(MpcDataset):
    rows: list[IdentificationDatasetRow]


class IdentificationDatasetSummary(MpcDatasetSummary):
    pass
