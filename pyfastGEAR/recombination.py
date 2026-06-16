from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .counts import create_baps_output_struct, update_counts_with_strain
from .generate_output import save_pickle
from .hmm import hmm_recombination_analysis, pre_calculate_trans_matrices
from .models import BapsData, UserData
from .segments import identify_segments


def learn_rho(segments_for_strains: list[np.ndarray], total_sequence_length: int) -> float:
    y1 = 0.0
    y2 = 0.0
    for segments in segments_for_strains:
        y1 += float(np.sum(segments[:, 1] - segments[:, 0]))
        y2 += float(segments.shape[0] - 1)
    return float((total_sequence_length - 2 + y1) / (total_sequence_length - 2 + y1 + y2))


def learn_trans_pars(
    segments_for_strains: list[np.ndarray],
    total_sequence_length: int,
    trans_pars: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(trans_pars)
    y3 = 0.0
    y4 = 0.0
    y5 = 0.0
    y6 = 0.0
    y7 = 0.0
    home = int(updated["home"])

    for segments in segments_for_strains:
        home_segments = segments[segments[:, 2] == home, :]
        if home_segments.size > 0:
            y3 += float(np.sum(home_segments[:, 1] - home_segments[:, 0]))
            y4 += float(np.sum(home_segments[:, 1] < total_sequence_length))

        rec_segments = segments[segments[:, 2] != home, :]
        if rec_segments.size > 0:
            y5 += float(np.sum(rec_segments[:, 1] - rec_segments[:, 0]))
            next_segment_origin = np.concatenate([segments[1:, 2], np.array([0], dtype=segments.dtype)])
            y6 += float(np.sum((segments[:, 2] != home) & (next_segment_origin == home)))
            y7 += float(np.sum((segments[:, 2] != home) & (next_segment_origin != home) & (next_segment_origin > 0)))

    updated["rho0"] = 0.5 ** (1.0 / (total_sequence_length - 1))
    if y6 + y7 <= 2:
        mean_tract_length = (float(updated["muMax"]) + float(updated["muMin"])) / 2.0
        updated["rho"] = (mean_tract_length - 1.0) / mean_tract_length
    else:
        updated["rho"] = y5 / (y5 + y6 + y7 - 2.0)

    updated["a"] = (y6 + float(updated["aPriorCount"]) - 1.0) / (
        y6 + y7 + float(updated["aPriorCount"]) + float(updated["negAPriorCount"]) - 2.0
    )
    return updated


def initialize_recent_recombination_baps(user_data: UserData, snp_data: np.ndarray) -> BapsData:
    grouped_partition = np.asarray(user_data.grouped_partition, dtype=np.int64)
    n_clusters = int(np.max(grouped_partition))
    return create_baps_output_struct(
        grouped_partition,
        snp_data,
        user_data.snp_positions,
        n_clusters + 1,
        None,
        None,
    )


def learn_recombinations_in_cluster(
    cluster_index: int,
    user_data: UserData,
    c: BapsData,
    output_dir: Path,
    reduced_output: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    strains_in_cluster = np.where(np.asarray(user_data.grouped_partition, dtype=np.int64) == cluster_index)[0] + 1
    n_strains_in_cluster = len(strains_in_cluster)
    n_snps = len(user_data.snp_positions)
    n_states = c.COUNTS.shape[2]

    if user_data.n_iter_completed == 0:
        cluster_trans_pars = dict(user_data.trans_pars)
        cluster_trans_pars["home"] = cluster_index
        simulated_pop_structure_for_cluster = np.full((n_strains_in_cluster, n_snps), cluster_index, dtype=np.uint8)
    else:
        raise NotImplementedError("Stage-6 resume behavior is deferred for the one-lineage slice.")

    segments_for_strains: list[np.ndarray] = [np.zeros((0, 3), dtype=np.int64) for _ in range(n_strains_in_cluster)]
    final_pop_structure_for_cluster = np.full((n_strains_in_cluster, n_snps), cluster_index, dtype=np.uint8)
    marginals_for_strains: list[np.ndarray | None] = [None] * n_strains_in_cluster
    rng = rng or np.random.default_rng()

    c_work = replace(c, SUMCOUNTS=c.SUMCOUNTS.copy(), COUNTS=c.COUNTS.copy())
    for iteration in range(user_data.n_iter_completed + 1, user_data.n_iter_completed + user_data.n_iter + 1):
        pre_calculated_trans_matrices = pre_calculate_trans_matrices(
            user_data.total_sequence_length,
            cluster_trans_pars,
            n_states,
            user_data.snp_positions,
        )

        for index_of_strain_in_cluster, strain_index in enumerate(strains_in_cluster):
            new_states: list[int] = []
            old_states = simulated_pop_structure_for_cluster[index_of_strain_in_cluster, :]
            change_this_site = old_states == cluster_index
            c_work = update_counts_with_strain(c_work, int(strain_index), new_states, old_states, change_this_site)

            marginals, _, simulated_states = hmm_recombination_analysis(
                c_work,
                pre_calculated_trans_matrices,
                strain_index=int(strain_index),
                rng=rng,
            )

            old_states = []
            new_states = simulated_states
            change_this_site = np.asarray(new_states) == cluster_index
            c_work = update_counts_with_strain(c_work, int(strain_index), new_states, old_states, change_this_site)

            simulated_pop_structure_for_cluster[index_of_strain_in_cluster, :] = np.asarray(simulated_states, dtype=np.uint8)
            segments_for_strains[index_of_strain_in_cluster] = identify_segments(
                simulated_pop_structure_for_cluster,
                index_of_strain_in_cluster + 1,
                c.snpPositions,
                user_data.total_sequence_length,
            )
            marginals_for_strains[index_of_strain_in_cluster] = marginals

            if iteration == user_data.n_iter_completed + user_data.n_iter:
                max_origin = np.argmax(marginals, axis=0) + 1
                recombinant_sites = np.where(marginals[cluster_index - 1, :] < user_data.probability_threshold)[0]
                final_pop_structure_for_cluster[index_of_strain_in_cluster, :] = cluster_index
                final_pop_structure_for_cluster[index_of_strain_in_cluster, recombinant_sites] = max_origin[recombinant_sites]

        cluster_trans_pars = learn_trans_pars(segments_for_strains, user_data.total_sequence_length, cluster_trans_pars)

    result = {
        "clusterIndex": cluster_index,
        "strainsInCluster": strains_in_cluster,
        "simulatedPopStructureForCluster": simulated_pop_structure_for_cluster,
        "segmentsForStrains": segments_for_strains,
        "clusterTransPars": cluster_trans_pars,
        "finalPopStructureForCluster": final_pop_structure_for_cluster,
        "marginalsForStrains": marginals_for_strains,
    }
    if not reduced_output:
        save_pickle(output_dir / f"lineage_{cluster_index}_recombination.pkl", result)
    return result


def learn_recombinations_in_clusters(
    user_data: UserData,
    snp_data: np.ndarray,
    output_dir: Path,
    reduced_output: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[UserData, list[dict[str, Any]]]:
    c = initialize_recent_recombination_baps(user_data, snp_data)
    n_snps = len(user_data.snp_positions)
    updated_pop_structure = np.tile(np.asarray(user_data.grouped_partition, dtype=np.uint8)[:, np.newaxis], (1, n_snps))
    n_clusters = int(np.max(np.asarray(user_data.grouped_partition, dtype=np.int64)))
    lineage_results: list[dict[str, Any]] = []
    rng = rng or np.random.default_rng()

    for cluster_index in range(1, n_clusters + 1):
        result = learn_recombinations_in_cluster(
            cluster_index,
            user_data,
            c,
            output_dir,
            reduced_output=reduced_output,
            rng=rng,
        )
        lineage_results.append(result)
        strains_in_cluster = np.asarray(result["strainsInCluster"], dtype=np.int64) - 1
        updated_pop_structure[strains_in_cluster, :] = np.asarray(result["finalPopStructureForCluster"], dtype=np.uint8)

    updated_user_data = replace(
        user_data,
        n_iter_completed=user_data.n_iter_completed + user_data.n_iter,
        state="recProfilesCalculated",
        updated_pop_structure=updated_pop_structure,
    )
    return updated_user_data, lineage_results


def learn_recombinations_in_clusters_one_lineage(
    user_data: UserData,
    snp_data: np.ndarray,
    cluster_index: int,
    output_dir: Path,
    reduced_output: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[UserData, dict[str, Any]]:
    c = initialize_recent_recombination_baps(user_data, snp_data)
    result = learn_recombinations_in_cluster(cluster_index, user_data, c, output_dir, reduced_output, rng=rng)

    updated_pop_structure = np.tile(np.asarray(user_data.grouped_partition, dtype=np.uint8)[:, np.newaxis], (1, len(user_data.snp_positions)))
    strains_in_cluster = np.asarray(result["strainsInCluster"], dtype=np.int64) - 1
    updated_pop_structure[strains_in_cluster, :] = np.asarray(result["finalPopStructureForCluster"], dtype=np.uint8)
    updated_user_data = replace(
        user_data,
        n_iter_completed=user_data.n_iter_completed + user_data.n_iter,
        state="recProfilesCalculated",
        updated_pop_structure=updated_pop_structure,
    )
    return updated_user_data, result
