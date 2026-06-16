from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .clustering import reorder_cluster_labels
from .counts import create_baps_output_struct
from .generate_output import save_pickle
from .hmm import hmm_recombination_analysis, pre_calculate_trans_matrices
from .models import UserData
from .segments import identify_segments


def inspect_cluster_relationships(
    user_data: UserData,
    snp_data: np.ndarray,
    output_dir: Path,
    analyse_lineages: bool = False,
) -> tuple[UserData, list[list[np.ndarray | None]]]:
    partition = reorder_cluster_labels(np.asarray(user_data.partition, dtype=np.int64))
    user_data = replace(user_data, partition=partition)

    if analyse_lineages:
        n_snps = len(user_data.snp_positions)
        masked_snp_data = np.asarray(snp_data).copy()
        structure_without_recent = np.tile(np.asarray(user_data.grouped_partition, dtype=np.uint8)[:, np.newaxis], (1, n_snps))
        recent_recombination_inds = structure_without_recent != np.asarray(user_data.updated_pop_structure, dtype=np.uint8)
        masked_snp_data[recent_recombination_inds] = 0
        partition_for_analysis = np.asarray(user_data.grouped_partition, dtype=np.int64)
        n_clusters = int(np.max(partition_for_analysis))
        c = create_baps_output_struct(
            partition_for_analysis,
            masked_snp_data,
            user_data.snp_positions,
            n_clusters,
            None,
            None,
        )
    else:
        n_clusters = int(np.max(partition))
        c = create_baps_output_struct(
            partition,
            snp_data,
            user_data.snp_positions,
            n_clusters,
            None,
            None,
        )
    phi = user_data.prior_prob_of_no_breaks
    pre_calculated_trans_matrices = pre_calculate_trans_matrices(
        user_data.total_sequence_length,
        phi,
        2,
        c.snpPositions,
        0,
    )

    all_pairwise_marginals: list[list[np.ndarray | None]] = [
        [None for _ in range(n_clusters)] for _ in range(n_clusters)
    ]
    for cluster1 in range(1, n_clusters):
        for cluster2 in range(cluster1 + 1, n_clusters + 1):
            marginals, _, _ = hmm_recombination_analysis(
                c,
                pre_calculated_trans_matrices,
                strain_index=None,
                cluster_indices=[cluster1, cluster2],
                rng=np.random.default_rng(1),
            )
            all_pairwise_marginals[cluster1 - 1][cluster2 - 1] = marginals

    prefix = "lineage_comparison" if analyse_lineages else "cluster_comparison"
    save_pickle(output_dir / f"{prefix}.pkl", {"allPairwiseMarginals": all_pairwise_marginals})
    return user_data, all_pairwise_marginals


def compute_cluster_distances(user_data: UserData, all_pairwise_marginals: list[list[np.ndarray | None]]) -> np.ndarray:
    n_clusters = int(np.max(np.asarray(user_data.partition, dtype=np.int64)))
    cluster_dist = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    for cluster1 in range(1, n_clusters):
        for cluster2 in range(cluster1 + 1, n_clusters + 1):
            marginals = all_pairwise_marginals[cluster1 - 1][cluster2 - 1]
            if marginals is None:
                continue
            is_same = marginals[0, :] > 0.5
            segments = identify_segments(
                np.asarray(is_same, dtype=np.uint8)[np.newaxis, :],
                1,
                user_data.snp_positions,
                user_data.total_sequence_length,
            )
            same_segments = segments[segments[:, 2] == 1, :]
            if same_segments.size > 0:
                total_length = np.sum(same_segments[:, 1] - same_segments[:, 0] + 1, axis=0)
                cluster_dist[cluster1 - 1, cluster2 - 1] = 1.0 - total_length / user_data.total_sequence_length
            else:
                cluster_dist[cluster1 - 1, cluster2 - 1] = 1.0
            cluster_dist[cluster2 - 1, cluster1 - 1] = cluster_dist[cluster1 - 1, cluster2 - 1]
    return cluster_dist


def create_cluster_groups(user_data: UserData) -> UserData:
    cluster_dist = np.asarray(user_data.cluster_dist, dtype=np.float64)
    n_clusters = cluster_dist.shape[0]
    if n_clusters == 1:
        cluster_groups = np.ones(1, dtype=np.int64)
    else:
        condensed = squareform(cluster_dist, checks=False)
        cluster_z = linkage(condensed, method="complete")
        cluster_groups = fcluster(cluster_z, t=0.5, criterion="distance").astype(np.int64)
    grouped_partition = cluster_groups[np.asarray(user_data.partition, dtype=np.int64) - 1]
    updated = replace(user_data, cluster_groups=cluster_groups, grouped_partition=grouped_partition)
    return reorder_cluster_groups(updated)


def lineage_similarities_to_lineage_structure(
    user_data: UserData,
    all_pairwise_marginals: list[list[np.ndarray | None]],
    type_index: int,
) -> UserData:
    if type_index == 1:
        partition = np.asarray(user_data.grouped_partition, dtype=np.int64)
        target_field = "lineage_structure"
    elif type_index == 2:
        partition = np.asarray(user_data.partition, dtype=np.int64)
        target_field = "cluster_structure"
    else:
        raise ValueError(f"Unsupported lineage structure type: {type_index}")

    n_components = int(np.max(partition))
    n_snps = len(user_data.snp_positions)
    structure = np.zeros((n_components, n_snps), dtype=np.uint8)
    for component_index in range(1, n_components):
        structure[component_index - 1, :] = component_index
        for join_target in range(n_components, component_index, -1):
            marginals = all_pairwise_marginals[component_index - 1][join_target - 1]
            if marginals is None:
                continue
            is_same = marginals[0, :] > 0.5
            sites_to_join = is_same & (structure[component_index - 1, :] == component_index)
            structure[component_index - 1, sites_to_join] = join_target
    structure[n_components - 1, :] = n_components

    if target_field == "lineage_structure":
        return replace(user_data, lineage_structure=structure)
    return replace(user_data, cluster_structure=structure)


def derive_pop_structure_from_lineage_structure(user_data: UserData, type_index: int) -> np.ndarray:
    if type_index == 2:
        structure = np.asarray(user_data.lineage_structure_cleaned)
        partition = np.asarray(user_data.grouped_partition, dtype=np.int64)
    elif type_index == 3:
        structure = np.asarray(user_data.cluster_structure_cleaned)
        partition = np.asarray(user_data.partition, dtype=np.int64)
    elif type_index == 4:
        structure = np.asarray(user_data.lineage_structure)
        partition = np.asarray(user_data.grouped_partition, dtype=np.int64)
    elif type_index == 5:
        structure = np.asarray(user_data.cluster_structure)
        partition = np.asarray(user_data.partition, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported pop structure type: {type_index}")

    n_components = int(np.max(partition))
    n_snps = len(user_data.snp_positions)
    pop_structure = np.zeros((partition.shape[0], n_snps), dtype=np.uint8)
    for component_index in range(1, structure.shape[0] + 1):
        strain_ids = np.where(partition == component_index)[0]
        merged_assignments = structure[component_index - 1, :].astype(np.uint8, copy=False)
        pop_structure[strain_ids, :] = np.tile(merged_assignments, (len(strain_ids), 1))
    return pop_structure


def calc_dist_matrix(partition_array: np.ndarray) -> np.ndarray:
    partition_array = np.asarray(partition_array)
    ninds, nloci = partition_array.shape
    dist_matrix = np.zeros((ninds, ninds), dtype=np.float64)
    for ind1 in range(ninds - 1):
        for ind2 in range(ind1 + 1, ninds):
            dist_matrix[ind1, ind2] = np.sum(partition_array[ind1, :] != partition_array[ind2, :]) / nloci
            dist_matrix[ind2, ind1] = dist_matrix[ind1, ind2]
    return dist_matrix


def reorder_cluster_groups(user_data: UserData) -> UserData:
    grouped_partition = np.asarray(user_data.grouped_partition, dtype=np.int64)
    cluster_groups = np.asarray(user_data.cluster_groups, dtype=np.int64)
    n_lineages = len(np.unique(grouped_partition))
    lineage_sizes = np.array([np.sum(grouped_partition == lineage_index) for lineage_index in range(1, n_lineages + 1)])
    ordered_lineages = np.arange(1, n_lineages + 1)[np.argsort(lineage_sizes, kind="stable")]

    new_partition = np.zeros_like(grouped_partition)
    cluster_partition = np.zeros_like(cluster_groups)
    for lineage_index, ordered_lineage in enumerate(ordered_lineages, start=1):
        new_partition[grouped_partition == ordered_lineage] = lineage_index
        cluster_partition[cluster_groups == ordered_lineage] = lineage_index
    return replace(user_data, grouped_partition=new_partition, cluster_groups=cluster_partition)


def run_pre_recombination_relationships(
    user_data: UserData,
    snp_data: np.ndarray,
    output_dir: Path,
) -> tuple[UserData, dict[str, Any]]:
    user_data = replace(user_data, output_dir=str(output_dir))
    user_data, all_pairwise_marginals = inspect_cluster_relationships(user_data, snp_data, output_dir, False)
    cluster_dist = compute_cluster_distances(user_data, all_pairwise_marginals)
    user_data = replace(user_data, cluster_dist=cluster_dist)
    user_data = create_cluster_groups(user_data)
    user_data = lineage_similarities_to_lineage_structure(user_data, all_pairwise_marginals, 2)
    comparison_payload = {"allPairwiseMarginals": all_pairwise_marginals}
    save_pickle(output_dir / "cluster_relationships.pkl", comparison_payload)
    return replace(user_data, state="clusterRelationshipsInspected"), comparison_payload


def run_post_recombination_relationships(
    user_data: UserData,
    snp_data: np.ndarray,
    output_dir: Path,
) -> tuple[UserData, dict[str, Any]]:
    user_data = replace(user_data, output_dir=str(output_dir))
    user_data, all_pairwise_marginals = inspect_cluster_relationships(user_data, snp_data, output_dir, True)
    user_data = lineage_similarities_to_lineage_structure(user_data, all_pairwise_marginals, 1)
    comparison_payload = {"allPairwiseMarginals": all_pairwise_marginals}
    save_pickle(output_dir / "lineage_relationships.pkl", comparison_payload)
    return replace(user_data, state="relationshipsReInspected"), comparison_payload
