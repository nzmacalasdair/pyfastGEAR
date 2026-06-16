from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.special import gammaln

from .models import UserData
from .segments import identify_segments


def create_ancestral_sequences(user_data: UserData, snp_data: np.ndarray, for_lineages: bool) -> np.ndarray:
    partition = np.asarray(user_data.grouped_partition if for_lineages else user_data.partition, dtype=np.int64)
    n_snps = len(user_data.snp_positions)
    n_clusters = int(np.max(partition))
    ancestral_seq = np.zeros((n_clusters, n_snps), dtype=np.int64)

    masked_snp_data = np.asarray(snp_data).copy()
    structure_without_recent = np.tile(np.asarray(user_data.grouped_partition, dtype=np.uint8)[:, np.newaxis], (1, n_snps))
    recent_recombination_inds = structure_without_recent != np.asarray(user_data.updated_pop_structure, dtype=np.uint8)
    masked_snp_data[recent_recombination_inds] = 9

    for cluster_index in range(1, n_clusters + 1):
        cluster_sequences = masked_snp_data[partition == cluster_index, :]
        for snp_index in range(n_snps):
            observations = cluster_sequences[:, snp_index]
            # MATLAB uses setdiff(observations, 9), which removes the
            # missing-code 9 and also de-duplicates/sorts observations.
            observations = np.setdiff1d(observations, np.array([9], dtype=observations.dtype))
            if observations.size == 0:
                ancestral_seq[cluster_index - 1, snp_index] = 9
            else:
                alleles, counts = np.unique(observations, return_counts=True)
                ancestral_seq[cluster_index - 1, snp_index] = int(alleles[np.argmax(counts)])
    return ancestral_seq.astype(np.uint8)


def clear_non_significant_recent_recombinations(user_data: UserData, snp_data: np.ndarray) -> UserData:
    if len(user_data.snp_positions) == 1:
        return replace(
            user_data,
            pop_structure_cleaned=np.asarray(user_data.updated_pop_structure, dtype=np.uint8),
            recent_recombination_significances=[],
        )

    ancestral_seq_table = create_ancestral_sequences(user_data, snp_data, True)
    n_strains = len(np.asarray(user_data.partition))
    pop_structure_cleaned = np.asarray(user_data.updated_pop_structure, dtype=np.uint8).copy()
    recent_recombination_significances: list[np.ndarray] = []

    for strain_index in range(1, n_strains + 1):
        segments = identify_segments(
            np.asarray(user_data.updated_pop_structure, dtype=np.uint8),
            strain_index,
            user_data.snp_positions,
            user_data.total_sequence_length,
        )
        home_lineage = int(np.asarray(user_data.grouped_partition, dtype=np.int64)[strain_index - 1])
        home_mask = segments[:, 2] == home_lineage
        home_segments = segments[home_mask, :]
        rec_segments = segments[~home_mask, :]

        if rec_segments.shape[0] > 0:
            home_snp_differences = formulate_snp_difference_between_strain_and_lineage(
                ancestral_seq_table,
                home_lineage,
                np.asarray(snp_data),
                strain_index,
                user_data.total_sequence_length,
                user_data.snp_positions,
            )
            home_counts = _compute_segment_snp_counts(home_segments, home_snp_differences["snp_positions"])
            home_counts = np.array([np.sum(home_counts[:, 0]), np.sum(home_counts[:, 1])], dtype=np.float64)
            rec_counts = _compute_segment_snp_counts(rec_segments, home_snp_differences["snp_positions"])
            log_bf = compute_log_bf_for_recombinations(rec_counts, home_counts, np.asarray(user_data.prior_counts, dtype=np.float64))
            is_combined = np.zeros(rec_segments.shape[0], dtype=bool)

            while np.any(log_bf <= 0):
                min_index = int(np.argmin(log_bf))
                is_combined[min_index] = True
                home_counts = home_counts + rec_counts[min_index, :]
                remaining = ~is_combined
                if np.any(remaining):
                    log_bf[remaining] = compute_log_bf_for_recombinations(
                        rec_counts[remaining, :],
                        home_counts,
                        np.asarray(user_data.prior_counts, dtype=np.float64),
                    )
                log_bf[is_combined] = 1e5
            cleaned = clean_strain_structure(
                np.asarray(user_data.updated_pop_structure, dtype=np.uint8)[strain_index - 1, :],
                rec_segments[is_combined, :],
                user_data.snp_positions,
                home_lineage,
            )
            pop_structure_cleaned[strain_index - 1, :] = cleaned
            recent_recombination_significances.append(np.asarray(log_bf[~is_combined], dtype=np.float64))
        else:
            recent_recombination_significances.append(np.zeros(0, dtype=np.float64))

    return replace(
        user_data,
        pop_structure_cleaned=pop_structure_cleaned,
        recent_recombination_significances=recent_recombination_significances,
    )


def clear_non_significant_ancestral_recombinations(
    user_data: UserData,
    snp_data: np.ndarray,
    type_index: int,
) -> UserData:
    if type_index == 1:
        ancestral_seq_table = create_ancestral_sequences(user_data, snp_data, True)
        n_lineages = int(np.max(np.asarray(user_data.grouped_partition, dtype=np.int64)))
        lineage_structure = np.asarray(user_data.lineage_structure, dtype=np.uint8)
    else:
        ancestral_seq_table = create_ancestral_sequences(user_data, snp_data, False)
        n_lineages = int(np.max(np.asarray(user_data.partition, dtype=np.int64)))
        lineage_structure = np.asarray(user_data.cluster_structure, dtype=np.uint8)

    structure_cleaned = lineage_structure.copy()
    ancestral_recombination_significances: list[np.ndarray] = [np.zeros(0, dtype=np.float64) for _ in range(n_lineages)]
    snp_differences_between_lineages: list[list[dict[str, Any] | None]] = [[None] * n_lineages for _ in range(n_lineages)]
    for lineage_index1 in range(1, n_lineages):
        for lineage_index2 in range(lineage_index1 + 1, n_lineages + 1):
            snp_differences_between_lineages[lineage_index1 - 1][lineage_index2 - 1] = formulate_snp_difference_data(
                ancestral_seq_table,
                lineage_index1,
                lineage_index2,
                user_data.total_sequence_length,
                user_data.snp_positions,
            )

    for lineage_index in range(1, n_lineages + 1):
        significance_results = clear_non_significant_ancestral_recombinations_in_lineage(
            snp_differences_between_lineages,
            lineage_index,
            lineage_structure,
            user_data.snp_positions,
            user_data.total_sequence_length,
            float(user_data.ancestral_log_bf_threshold),
            np.asarray(user_data.prior_counts, dtype=np.float64),
        )
        if significance_results["segments"].shape[0] == 1:
            significance_results["segments"][0, 2] = lineage_index
        structure_cleaned[lineage_index - 1, :] = infer_lineage_structure(
            user_data.snp_positions,
            significance_results["segments"],
            lineage_index,
            significance_results["log_bf_for_keeping_separate"],
            float(user_data.ancestral_log_bf_threshold),
        )
        mask = (significance_results["segments"][:, 2] != lineage_index) & (
            significance_results["log_bf_for_keeping_separate"] > float(user_data.ancestral_log_bf_threshold)
        )
        ancestral_recombination_significances[lineage_index - 1] = np.asarray(
            significance_results["log_bf_for_keeping_separate"][mask],
            dtype=np.float64,
        )

    if type_index == 1:
        return replace(
            user_data,
            lineage_structure_cleaned=structure_cleaned.astype(np.uint8),
            lineage_ancestral_log_bf=ancestral_recombination_significances,
        )
    return replace(
        user_data,
        cluster_structure_cleaned=structure_cleaned.astype(np.uint8),
        cluster_ancestral_log_bf=ancestral_recombination_significances,
    )


def clean_strain_structure(orig_structure: np.ndarray, segments_to_clean: np.ndarray, snp_positions: np.ndarray, home_lineage: int) -> np.ndarray:
    cleaned = np.asarray(orig_structure, dtype=np.uint8).copy()
    for first, last, _ in np.asarray(segments_to_clean, dtype=np.int64):
        cleaned[(snp_positions >= first) & (snp_positions <= last)] = home_lineage
    return cleaned


def compute_log_bf_for_recombinations(segment_counts: np.ndarray, home_counts: np.ndarray, prior_counts: np.ndarray) -> np.ndarray:
    return np.array([compute_binomial_logml(np.vstack([segment_count, home_counts]), prior_counts) for segment_count in segment_counts], dtype=np.float64)


def formulate_snp_difference_between_strain_and_lineage(
    ancestral_seq_table: np.ndarray,
    lineage_index: int,
    snp_data: np.ndarray,
    strain_index: int,
    total_sequence_length: int,
    snp_positions: np.ndarray,
) -> dict[str, Any]:
    seq_for_comparison = np.vstack([ancestral_seq_table[lineage_index - 1, :], snp_data[strain_index - 1, :]])
    n_alleles = np.array([len(np.unique(seq_for_comparison[:, i])) for i in range(seq_for_comparison.shape[1])], dtype=np.int64)
    either_is_unknown = np.any((seq_for_comparison == 9) | (seq_for_comparison == 0), axis=0)
    n_alleles[either_is_unknown] = 1
    return {
        "total_sequence_length": total_sequence_length,
        "snp_positions": np.asarray(snp_positions)[n_alleles != 1],
    }


def compute_binomial_logml(counts: np.ndarray, prior_counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.vstack([counts, np.sum(counts, axis=0)])
    prior = np.tile(np.asarray(prior_counts, dtype=np.float64), (3, 1))
    logml = (
        gammaln(np.sum(prior, axis=1))
        - np.sum(gammaln(prior), axis=1)
        + np.sum(gammaln(prior + counts), axis=1)
        - gammaln(np.sum(prior + counts, axis=1))
    )
    return float(logml[0] + logml[1] - logml[2])


def infer_lineage_structure(
    snp_positions: np.ndarray,
    segments: np.ndarray,
    lineage_index: int,
    log_bf_for_keeping_separate: np.ndarray,
    ancestral_log_bf_threshold: float,
) -> np.ndarray:
    segments = np.asarray(segments, dtype=np.int64).copy()
    non_significant = np.where((segments[:, 2] != lineage_index) & (log_bf_for_keeping_separate < ancestral_log_bf_threshold))[0]
    segments[non_significant, 2] = lineage_index
    lineage_structure = np.zeros(len(snp_positions), dtype=np.uint8)
    for first, last, origin in segments:
        lineage_structure[(snp_positions >= first) & (snp_positions <= last)] = origin
    if np.any(lineage_structure == 0):
        raise ValueError("Lineage structure contains unassigned sites")
    return lineage_structure


def clear_non_significant_ancestral_recombinations_in_lineage(
    snp_differences_between_lineages: list[list[dict[str, Any] | None]],
    lineage_index: int,
    lineage_structure: np.ndarray,
    snp_positions: np.ndarray,
    total_sequence_length: int,
    ancestral_log_bf_threshold: float,
    prior_counts: np.ndarray,
) -> dict[str, Any]:
    segments = identify_segments(np.asarray(lineage_structure, dtype=np.uint8), lineage_index, snp_positions, total_sequence_length)
    n_segments = segments.shape[0]
    if n_segments == 1:
        return {
            "segments": segments,
            "neighbor_to_join_with": np.array([0], dtype=np.float64),
            "log_bf_for_keeping_separate": np.array([np.nan], dtype=np.float64),
        }

    segment_lengths = segments[:, 1] - segments[:, 0] + 1
    log_bf = np.zeros(n_segments, dtype=np.float64)
    neighbor_to_join_with = np.zeros(n_segments, dtype=np.float64)
    for segment_index in range(1, n_segments + 1):
        results = compute_significances(snp_differences_between_lineages, segments, segment_index, prior_counts, lineage_index)
        if segments[segment_index - 1, 2] != lineage_index:
            log_bf[segment_index - 1] = float(results)
            neighbor_to_join_with[segment_index - 1] = -1
        else:
            results_arr = np.atleast_1d(np.asarray(results, dtype=np.float64))
            min_index = int(np.argmin(results_arr))
            min_value = float(results_arr[min_index])
            if results_arr.size == 1:
                log_bf[segment_index - 1] = min_value
                neighbor_to_join_with[segment_index - 1] = 0
            else:
                neighbor_lengths = segment_lengths[[segment_index - 2, segment_index]]
                if (
                    segment_lengths[segment_index - 1] > neighbor_lengths[min_index]
                    and segment_lengths[segment_index - 1] <= neighbor_lengths[1 - min_index]
                    and results_arr[1 - min_index] < ancestral_log_bf_threshold
                ):
                    neighbor_to_join_with[segment_index - 1] = 2 - min_index
                    log_bf[segment_index - 1] = float(results_arr[1 - min_index])
                else:
                    neighbor_to_join_with[segment_index - 1] = min_index + 1
                    log_bf[segment_index - 1] = min_value

    ind_to_merge = determine_index_to_merge_with_neighbors(segments, log_bf, neighbor_to_join_with, ancestral_log_bf_threshold)
    while not np.isnan(ind_to_merge):
        idx = int(ind_to_merge)
        segments = merge_segment_with_neighbors(segments, idx, int(neighbor_to_join_with[idx - 1]), lineage_index)
        neighbor_to_join_with, log_bf = update_significances(
            snp_differences_between_lineages,
            segments,
            idx,
            neighbor_to_join_with,
            log_bf,
            prior_counts,
            lineage_index,
            ancestral_log_bf_threshold,
        )
        ind_to_merge = determine_index_to_merge_with_neighbors(segments, log_bf, neighbor_to_join_with, ancestral_log_bf_threshold)

    return {
        "segments": segments,
        "neighbor_to_join_with": neighbor_to_join_with,
        "log_bf_for_keeping_separate": log_bf,
    }


def update_significances(
    snp_differences_between_lineages: list[list[dict[str, Any] | None]],
    segments: np.ndarray,
    joined_index: int,
    neighbor_to_join_with: np.ndarray,
    log_bf_for_keeping_separate: np.ndarray,
    prior_counts: np.ndarray,
    lineage_index: int,
    ancestral_log_bf_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_segments_before = len(log_bf_for_keeping_separate)
    neighbor_val = int(neighbor_to_join_with[joined_index - 1])
    if neighbor_val == 1:
        to_remove = [joined_index]
        to_set_to_nan = joined_index - 1
    elif neighbor_val == 2:
        to_remove = [joined_index + 1]
        to_set_to_nan = joined_index
    elif neighbor_val == 0:
        if joined_index == 1:
            to_remove = [2]
            to_set_to_nan = 1
        elif joined_index == n_segments_before:
            to_remove = [n_segments_before]
            to_set_to_nan = n_segments_before - 1
        else:
            to_remove = [joined_index, joined_index + 1]
            to_set_to_nan = joined_index - 1
    else:
        left_home = joined_index > 1 and int(neighbor_to_join_with[joined_index - 2]) != -1
        right_home = joined_index < n_segments_before and int(neighbor_to_join_with[joined_index]) != -1
        if joined_index > 1 and joined_index < n_segments_before and left_home and right_home:
            to_remove = [joined_index, joined_index + 1]
            to_set_to_nan = joined_index - 1
        elif joined_index > 1 and left_home:
            to_remove = [joined_index]
            to_set_to_nan = joined_index - 1
        elif joined_index < n_segments_before and right_home:
            to_remove = [joined_index + 1]
            to_set_to_nan = joined_index
        else:
            to_remove = []
            to_set_to_nan = joined_index

    keep_mask = np.ones(len(log_bf_for_keeping_separate), dtype=bool)
    for idx in to_remove:
        keep_mask[idx - 1] = False
    log_bf = log_bf_for_keeping_separate[keep_mask]
    neighbors = neighbor_to_join_with[keep_mask]
    log_bf[to_set_to_nan - 1] = np.nan
    neighbors[to_set_to_nan - 1] = np.nan

    n_segments = segments.shape[0]
    if n_segments == 1:
        return np.array([0.0]), np.array([np.nan])

    new_segment_index = int(np.where(np.isnan(log_bf))[0][0]) + 1
    if new_segment_index == n_segments:
        segments_to_recompute = [new_segment_index - 1, new_segment_index]
    elif new_segment_index == 1:
        segments_to_recompute = [new_segment_index, new_segment_index + 1]
    else:
        segments_to_recompute = [new_segment_index - 1, new_segment_index, new_segment_index + 1]
    recombination_segments = list(np.where(segments[:, 2] != lineage_index)[0] + 1)
    segments_to_recompute = sorted(set(segments_to_recompute + recombination_segments))
    segment_lengths = segments[:, 1] - segments[:, 0] + 1

    for segment_index_now in segments_to_recompute:
        results = compute_significances(snp_differences_between_lineages, segments, segment_index_now, prior_counts, lineage_index)
        if segments[segment_index_now - 1, 2] != lineage_index:
            log_bf[segment_index_now - 1] = float(results)
            neighbors[segment_index_now - 1] = -1
        else:
            results_arr = np.atleast_1d(np.asarray(results, dtype=np.float64))
            min_index = int(np.argmin(results_arr))
            min_value = float(results_arr[min_index])
            if results_arr.size == 1:
                log_bf[segment_index_now - 1] = min_value
                neighbors[segment_index_now - 1] = 0
            else:
                neighbor_lengths = segment_lengths[[segment_index_now - 2, segment_index_now]]
                if (
                    segment_lengths[segment_index_now - 1] > neighbor_lengths[min_index]
                    and segment_lengths[segment_index_now - 1] <= neighbor_lengths[1 - min_index]
                    and results_arr[1 - min_index] < ancestral_log_bf_threshold
                ):
                    neighbors[segment_index_now - 1] = 2 - min_index
                    log_bf[segment_index_now - 1] = float(results_arr[1 - min_index])
                else:
                    neighbors[segment_index_now - 1] = min_index + 1
                    log_bf[segment_index_now - 1] = min_value
    return neighbors, log_bf


def determine_index_to_merge_with_neighbors(
    segments: np.ndarray,
    log_bf_for_keeping_separate: np.ndarray,
    neighbor_to_join_with: np.ndarray,
    log_bf_threshold: float,
) -> float:
    home_segment_indices = np.where(neighbor_to_join_with != -1)[0] + 1
    ordered_segments = np.argsort(log_bf_for_keeping_separate) + 1
    n_segments = segments.shape[0]
    segment_lengths = segments[:, 1] - segments[:, 0] + 1
    if n_segments <= 1:
        return np.nan
    for index_to_try in ordered_segments:
        neighbor_val = int(neighbor_to_join_with[index_to_try - 1])
        if neighbor_val == 0:
            if index_to_try == 1:
                neighbor_length = segment_lengths[1]
            elif index_to_try == n_segments:
                neighbor_length = segment_lengths[-2]
            else:
                neighbor_length = segment_lengths[index_to_try - 2] + segment_lengths[index_to_try]
        elif neighbor_val == 1:
            neighbor_length = segment_lengths[index_to_try - 2]
        elif neighbor_val == 2:
            neighbor_length = segment_lengths[index_to_try]
        else:
            neighbor_length = int(np.sum(segment_lengths[np.asarray(home_segment_indices, dtype=np.int64) - 1]))
        if log_bf_for_keeping_separate[index_to_try - 1] < log_bf_threshold and neighbor_length >= segment_lengths[index_to_try - 1]:
            return float(index_to_try)
    return np.nan


def merge_segment_with_neighbors(segments: np.ndarray, segment_index: int, neighbor_to_join_with: int, lineage_index: int) -> np.ndarray:
    segments = np.asarray(segments, dtype=np.int64).copy()
    n_segments = segments.shape[0]
    if neighbor_to_join_with == 1:
        neighbor_index: list[int] = [segment_index - 1]
    elif neighbor_to_join_with == 2:
        neighbor_index = [segment_index + 1]
    elif neighbor_to_join_with == 0:
        if segment_index == 1:
            neighbor_index = [2]
        elif segment_index == n_segments:
            neighbor_index = [n_segments - 1]
        else:
            neighbor_index = [segment_index - 1, segment_index + 1]
    else:
        neighbor_index = []
        if segment_index > 1 and segments[segment_index - 2, 2] == lineage_index:
            neighbor_index.append(segment_index - 1)
        if segment_index < n_segments and segments[segment_index, 2] == lineage_index:
            neighbor_index.append(segment_index + 1)

    if len(neighbor_index) == 1:
        idx = neighbor_index[0]
        if idx == segment_index - 1:
            segments[idx - 1, 1] = segments[segment_index - 1, 1]
        else:
            segments[idx - 1, 0] = segments[segment_index - 1, 0]
        segments = np.delete(segments, segment_index - 1, axis=0)
    elif len(neighbor_index) == 2:
        segments[neighbor_index[0] - 1, 1] = segments[neighbor_index[1] - 1, 1]
        segments = np.delete(segments, [segment_index - 1, neighbor_index[1] - 1], axis=0)
    else:
        segments[segment_index - 1, 2] = lineage_index
    return segments


def compute_significances(
    snp_differences_between_lineages: list[list[dict[str, Any] | None]],
    segments: np.ndarray,
    segment_index: int,
    prior_counts: np.ndarray,
    lineage_index: int,
) -> float | np.ndarray:
    current_origin = int(segments[segment_index - 1, 2])
    n_segments = segments.shape[0]
    if n_segments == 1:
        raise ValueError("Number of segments is one")
    if current_origin == lineage_index:
        left_origin = None if segment_index == 1 else int(segments[segment_index - 2, 2])
        right_origin = None if segment_index == n_segments else int(segments[segment_index, 2])
        if left_origin is None:
            alt_origin: int | list[int] = right_origin  # type: ignore[assignment]
            neighbor_segments = [segment_index + 1]
        elif right_origin is None:
            alt_origin = left_origin
            neighbor_segments = [segment_index - 1]
        elif right_origin == left_origin:
            alt_origin = left_origin
            neighbor_segments = [segment_index - 1, segment_index + 1]
        else:
            alt_origin = [left_origin, right_origin]
            neighbor_segments = [segment_index - 1, segment_index + 1]
        if isinstance(alt_origin, list):
            return np.array(
                [
                    compute_log_bf_from_counts(
                        snp_differences_between_lineages,
                        current_origin,
                        alt_origin[i],
                        segments,
                        segment_index,
                        [neighbor_segments[i]],
                        prior_counts,
                    )
                    for i in range(2)
                ],
                dtype=np.float64,
            )
        return compute_log_bf_from_counts(
            snp_differences_between_lineages,
            current_origin,
            int(alt_origin),
            segments,
            segment_index,
            neighbor_segments,
            prior_counts,
        )
    return compute_log_bf_for_discarding_recombination(
        snp_differences_between_lineages,
        lineage_index,
        segments,
        segment_index,
        prior_counts,
    )


def compute_log_bf_for_discarding_recombination(
    snp_differences_between_lineages: list[list[dict[str, Any] | None]],
    home_lineage_index: int,
    segments: np.ndarray,
    segment_index: int,
    prior_counts: np.ndarray,
) -> float:
    current_origin = int(segments[segment_index - 1, 2])
    snp_difference_data = get_snp_difference_data(snp_differences_between_lineages, current_origin, home_lineage_index)
    segment_counts = get_snp_counts(snp_difference_data, segments, [segment_index])[0, :]
    home_segment_indices = list(np.where(segments[:, 2] == home_lineage_index)[0] + 1)
    home_lineage_counts = np.sum(get_snp_counts(snp_difference_data, segments, home_segment_indices), axis=0)
    return compute_binomial_logml(np.vstack([segment_counts, home_lineage_counts]), prior_counts)


def compute_log_bf_from_counts(
    snp_differences_between_lineages: list[list[dict[str, Any] | None]],
    current_origin: int,
    alt_origin: int,
    segments: np.ndarray,
    segment_index: int,
    neighbor_segments_to_consider: list[int],
    prior_counts: np.ndarray,
) -> float:
    snp_difference_data = get_snp_difference_data(snp_differences_between_lineages, current_origin, alt_origin)
    segment_counts = get_snp_counts(snp_difference_data, segments, [segment_index])[0, :]
    neighbor_segment_counts = np.sum(get_snp_counts(snp_difference_data, segments, neighbor_segments_to_consider), axis=0)
    return compute_binomial_logml(np.vstack([segment_counts, neighbor_segment_counts]), prior_counts)


def formulate_snp_difference_data(
    ancestral_seq_table: np.ndarray,
    current_origin: int,
    alt_origin: int,
    total_sequence_length: int,
    snp_positions: np.ndarray,
) -> dict[str, Any]:
    ancestral_seq_for_comparison = ancestral_seq_table[[current_origin - 1, alt_origin - 1], :]
    n_alleles = np.array([len(np.unique(ancestral_seq_for_comparison[:, i])) for i in range(ancestral_seq_for_comparison.shape[1])], dtype=np.int64)
    either_is_unknown = np.any(ancestral_seq_for_comparison == 9, axis=0)
    n_alleles[either_is_unknown] = 1
    return {
        "total_sequence_length": total_sequence_length,
        "snp_positions": np.asarray(snp_positions)[n_alleles != 1],
    }


def get_snp_difference_data(
    snp_differences_between_lineages: list[list[dict[str, Any] | None]],
    current_origin: int,
    alt_origin: int,
) -> dict[str, Any]:
    return snp_differences_between_lineages[min(current_origin, alt_origin) - 1][max(current_origin, alt_origin) - 1]  # type: ignore[index]


def get_snp_counts(data: dict[str, Any], segments: np.ndarray, segment_indices: list[int]) -> np.ndarray:
    counts = np.zeros((len(segment_indices), 2), dtype=np.float64)
    diff_positions = np.asarray(data["snp_positions"], dtype=np.int64)
    for row_index, segment_index in enumerate(segment_indices):
        first = int(segments[segment_index - 1, 0])
        last = int(segments[segment_index - 1, 1])
        counts[row_index, 0] = np.sum((diff_positions >= first) & (diff_positions <= last))
    segment_lengths = segments[np.asarray(segment_indices, dtype=np.int64) - 1, 1] - segments[np.asarray(segment_indices, dtype=np.int64) - 1, 0] + 1
    counts[:, 1] = segment_lengths - counts[:, 0]
    return counts


def _compute_segment_snp_counts(segments: np.ndarray, snp_positions: np.ndarray) -> np.ndarray:
    counts = np.zeros((segments.shape[0], 2), dtype=np.float64)
    for idx, (first, last, _) in enumerate(np.asarray(segments, dtype=np.int64)):
        n_snps = np.sum((snp_positions >= first) & (snp_positions <= last))
        counts[idx, 0] = n_snps
        counts[idx, 1] = (last - first + 1) - n_snps
    return counts
