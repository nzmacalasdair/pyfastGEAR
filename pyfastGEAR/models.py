from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class FastGearConfig:
    input_file: Path
    output_file: Path
    input_specs_file: Path | None = None
    partition_file: str = "-"
    n_iterations: int = 15
    n_clust_list: list[int] = field(default_factory=lambda: [10, 15, 20, 30])
    run_all_upper_bounds: bool = False
    reduced_output: bool = True
    output_dir: Path | None = None
    run_clustering: bool = False
    run_relationships: bool = False
    run_recombination: bool = False
    run_significance: bool = False
    write_outputs: bool = False
    run_recombination_lineage: int | None = None

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return self.output_file.resolve().parent / "output"


@dataclass(slots=True)
class PreprocessingResult:
    strain_labels: list[str]
    snp_data: np.ndarray
    snp_positions: np.ndarray
    total_sequence_length: int

    @property
    def n_strains(self) -> int:
        return len(self.strain_labels)

    @property
    def n_snps(self) -> int:
        return int(self.snp_positions.size)


@dataclass(slots=True)
class UserData:
    n_iter: int
    n_iter_completed: int
    strain_labels: list[str]
    snp_positions: np.ndarray
    total_sequence_length: int
    snp_data_file_name: str
    delimiter: str
    state: str
    prior_prob_of_no_breaks: float = 0.5
    prior_counts: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0], dtype=float))
    ancestral_log_bf_threshold: float = float(np.log(10.0))
    probability_threshold: float = 0.05
    alpha: float = 1.0
    trans_pars: dict[str, Any] = field(default_factory=dict)
    partition: np.ndarray | None = None
    grouped_partition: np.ndarray | None = None
    cluster_dist: np.ndarray | None = None
    cluster_groups: np.ndarray | None = None
    cluster_structure: np.ndarray | None = None
    lineage_structure: np.ndarray | None = None
    lineage_structure_cleaned: np.ndarray | None = None
    cluster_structure_cleaned: np.ndarray | None = None
    output_dir: str | None = None
    updated_pop_structure: np.ndarray | None = None
    pop_structure_cleaned: np.ndarray | None = None
    recent_recombination_significances: list[np.ndarray] | None = None
    lineage_ancestral_log_bf: list[np.ndarray] | None = None
    cluster_ancestral_log_bf: list[np.ndarray] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BapsData:
    PARTITION: np.ndarray
    npops: int
    data: np.ndarray
    noalle: np.ndarray
    snpPositions: np.ndarray
    SUMCOUNTS: np.ndarray
    COUNTS: np.ndarray
    adjprior: np.ndarray
    priorTerm: float
    rowsFromInd: int | None = None
    rows: np.ndarray | None = None
    Z: np.ndarray | None = None
    dist: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
