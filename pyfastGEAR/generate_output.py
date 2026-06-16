from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .models import FastGearConfig
from .segments import identify_segments


def infer_output_dir(output_file: Path) -> Path:
    return output_file.resolve().parent / "output"


def ensure_output_dir(config: FastGearConfig) -> Path:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_partition(user_data: Any, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    partition = np.asarray(user_data.partition, dtype=np.int64)
    grouped_partition = np.asarray(user_data.grouped_partition, dtype=np.int64)
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("StrainIndex  Lineage  Cluster  Name\n")
        for strain_index in range(1, len(partition) + 1):
            handle.write(
                f"{strain_index:<13.0f}{grouped_partition[strain_index-1]:<9.0f}{partition[strain_index-1]:<9.0f}{user_data.strain_labels[strain_index-1]}\n"
            )


def count_recent_recombinations(user_data: Any) -> dict[str, Any]:
    if len(user_data.snp_positions) == 1:
        return {"nRecent": 0}
    partition = np.asarray(user_data.partition, dtype=np.int64)
    grouped_partition = np.asarray(user_data.grouped_partition, dtype=np.int64)
    pop_structure_cleaned = np.asarray(user_data.pop_structure_cleaned, dtype=np.uint8)
    all_segments: list[np.ndarray] = []
    for strain_index in range(1, len(partition) + 1):
        segments = identify_segments(pop_structure_cleaned, strain_index, user_data.snp_positions, user_data.total_sequence_length)
        if segments.size == 0:
            continue
        extra = np.column_stack(
            [
                segments,
                np.full(segments.shape[0], grouped_partition[strain_index - 1], dtype=np.int64),
                np.full(segments.shape[0], strain_index, dtype=np.int64),
            ]
        )
        all_segments.append(extra)
    if not all_segments:
        return {"nRecent": 0}
    segments = np.vstack(all_segments)
    segments = segments[segments[:, 2] != segments[:, 3], :]
    n_lineages = int(np.max(grouped_partition))
    n_events = 0
    for lineage_index in range(1, n_lineages + 1):
        lineage_segments = segments[segments[:, 3] == lineage_index, :]
        if lineage_segments.size:
            n_events += _count_unique_recombinations_in_lineage(lineage_segments)
    return {"nRecent": int(n_events)}


def write_recombinations_to_text_file(user_data: Any, output_file: Path, type_index: int) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    n_snps = len(user_data.snp_positions)
    if n_snps == 1:
        pop_structure_cleaned = np.zeros((0, 0), dtype=np.uint8)
        lineage_structure_cleaned = np.zeros((0, 0), dtype=np.uint8)
        lineage_ancestral_log_bf: list[np.ndarray] = []
    else:
        pop_structure_cleaned = np.asarray(user_data.pop_structure_cleaned, dtype=np.uint8)
        lineage_structure_cleaned = np.asarray(user_data.lineage_structure_cleaned, dtype=np.uint8)
        lineage_ancestral_log_bf = user_data.lineage_ancestral_log_bf or []

    if type_index == 1:
        pop_structure = pop_structure_cleaned
        header_end = "STRAINS"
        first_row = "DonorLineage    RecipientStrain  log(BF)    StrainName"
    elif type_index == 2:
        pop_structure = lineage_structure_cleaned
        log_bf = user_data.lineage_ancestral_log_bf or []
        header_end = "LINEAGES"
        first_row = "Lineage1        Lineage2         log(BF)"
    else:
        raise ValueError(f"Unsupported output type: {type_index}")

    if type_index == 1:
        header = f"{count_recent_recombinations(user_data)['nRecent']} RECENT RECOMBINATION EVENTS"
    else:
        header = f"RECOMBINATIONS IN {header_end}"
    first_row = f"Start     End       {first_row}"

    with output_file.open("w", encoding="utf-8") as handle:
        handle.write(f"{header}\n")
        handle.write(f"{first_row}\n")
        if n_snps == 1:
            return
        n_rows = pop_structure.shape[0]
        for row_index in range(1, n_rows + 1):
            segments = identify_segments(pop_structure, row_index, user_data.snp_positions, user_data.total_sequence_length)
            home_cluster = int(np.asarray(user_data.grouped_partition, dtype=np.int64)[row_index - 1]) if type_index == 1 else row_index
            segments = segments[segments[:, 2] != home_cluster, :]
            n_segments = segments.shape[0]
            if n_segments == 0:
                continue
            table = np.column_stack([segments, np.full(n_segments, row_index, dtype=np.int64)])
            if type_index == 1:
                bf_values = _as_1d_float_array((user_data.recent_recombination_significances or [])[row_index - 1])
                table = np.column_stack([table, bf_values])
                strain_name = user_data.strain_labels[row_index - 1]
                for start, end, donor, recipient, bf in table:
                    handle.write(f"{start:<10.0f}{end:<10.0f}{donor:<16.0f}{recipient:<17.0f}{bf:<11.1f}{strain_name}\n")
            else:
                bf_values = _as_1d_float_array(log_bf[row_index - 1])
                if n_segments != len(bf_values):
                    raise ValueError("Incorrect number of significance scores for lineage recombinations")
                table = np.column_stack([table, bf_values])
                for start, end, donor, recipient, bf in table:
                    handle.write(f"{start:<10.0f}{end:<10.0f}{donor:<16.0f}{recipient:<17.0f}{bf:<10.1f}\n")


def write_outputs(user_data: Any, output_dir: Path) -> None:
    write_partition(user_data, output_dir / "lineage_information.txt")
    write_recombinations_to_text_file(user_data, output_dir / "recombinations_recent.txt", 1)
    write_recombinations_to_text_file(user_data, output_dir / "recombinations_ancestral.txt", 2)


def _count_unique_recombinations_in_lineage(segments: np.ndarray) -> int:
    segments = segments[np.argsort(segments[:, 0], kind="stable")]
    n_segments = segments.shape[0]
    event_labels = np.zeros(n_segments, dtype=np.int64)
    while np.any(event_labels == 0):
        to_process = int(np.where(event_labels == 0)[0][0])
        first, last, origin = segments[to_process, 0], segments[to_process, 1], segments[to_process, 2]
        overlapping = np.where(
            (
                ((segments[:, 0] < first) & (segments[:, 1] >= first))
                | ((segments[:, 1] > last) & (segments[:, 0] <= last))
                | ((segments[:, 0] >= first) & (segments[:, 1] <= last))
            )
            & (segments[:, 2] == origin)
        )[0]
        existing = np.setdiff1d(np.unique(event_labels[overlapping]), np.array([0], dtype=np.int64))
        if existing.size:
            event_labels[overlapping] = existing[0]
        else:
            event_labels[overlapping] = int(np.max(event_labels)) + 1
    return int(np.max(event_labels))


def _as_1d_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return arr.reshape(1)
    return arr.reshape(-1)
