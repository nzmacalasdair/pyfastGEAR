from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import gammaln

from .models import BapsData


def relabel_codes(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nloci = data.shape[1]
    noalle = np.zeros(nloci, dtype=np.int64)
    data_new = np.zeros(data.shape, dtype=np.uint8)

    for locus_index in range(nloci):
        orig_labels = np.unique(data[:, locus_index])
        orig_labels = orig_labels[orig_labels > 0]
        noalle[locus_index] = orig_labels.size
        for allele_index, orig_label in enumerate(orig_labels, start=1):
            data_new[data[:, locus_index] == orig_label, locus_index] = allele_index

    return data_new, noalle


def initial_counts(
    partition: np.ndarray,
    data: np.ndarray,
    npops: int,
    noalle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nloci = data.shape[1]
    max_noalle = int(np.max(noalle, initial=0))
    counts = np.zeros((max_noalle, nloci, npops), dtype=np.float64)
    sumcounts = np.zeros((npops, nloci), dtype=np.float64)

    for pop_index in range(1, npops + 1):
        for locus_index in range(nloci):
            observations = np.where((partition == pop_index) & (data[:, locus_index] > 0))[0]
            sumcounts[pop_index - 1, locus_index] = observations.size
            for allele_code in range(1, int(noalle[locus_index]) + 1):
                counts[allele_code - 1, locus_index, pop_index - 1] = np.sum(
                    data[observations, locus_index] == allele_code
                )

    return sumcounts, counts


def initial_counts_with_pop_structure(
    pop_structure: np.ndarray,
    snp_data: np.ndarray,
    noalle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_snps = snp_data.shape[1]
    n_pops = int(np.max(pop_structure))
    max_noalle = int(np.max(noalle, initial=0))
    counts = np.zeros((max_noalle, n_snps, n_pops), dtype=np.float64)
    sumcounts = np.zeros((n_pops, n_snps), dtype=np.float64)

    for locus_index in range(n_snps):
        for pop_index in range(1, n_pops + 1):
            observations = np.where(
                (pop_structure[:, locus_index] == pop_index) & (snp_data[:, locus_index] > 0)
            )[0]
            sumcounts[pop_index - 1, locus_index] = observations.size
            for allele_code in range(1, int(noalle[locus_index]) + 1):
                counts[allele_code - 1, locus_index, pop_index - 1] = np.sum(
                    snp_data[observations, locus_index] == allele_code
                )

    return sumcounts, counts


def initial_counts_with_origins(
    partition_array: np.ndarray,
    block_width: int,
    snp_data: np.ndarray,
    noalle: np.ndarray,
    snp_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_snps = snp_data.shape[1]
    n_pops = int(np.max(partition_array))
    n_intervals = partition_array.shape[1]
    max_noalle = int(np.max(noalle, initial=0))
    counts = np.zeros((max_noalle, n_snps, n_pops), dtype=np.float64)
    sumcounts = np.zeros((n_pops, n_snps), dtype=np.float64)

    for interval_index in range(n_intervals):
        interval_first_site = interval_index * block_width + 1
        interval_last_site = (interval_index + 1) * block_width
        snps_in_interval = np.where(
            (snp_positions >= interval_first_site) & (snp_positions <= interval_last_site)
        )[0]

        if snps_in_interval.size == 0:
            continue

        data_in_interval = snp_data[:, snps_in_interval]
        noalle_in_interval = noalle[snps_in_interval]
        interval_sumcounts, interval_counts = initial_counts(
            partition_array[:, interval_index],
            data_in_interval,
            n_pops,
            noalle_in_interval,
        )

        allele_padding = max_noalle - interval_counts.shape[0]
        if allele_padding > 0:
            interval_counts = np.concatenate(
                [
                    interval_counts,
                    np.zeros((allele_padding, snps_in_interval.size, n_pops), dtype=np.float64),
                ],
                axis=0,
            )

        counts[:, snps_in_interval, :] = interval_counts
        sumcounts[:, snps_in_interval] = interval_sumcounts

    return sumcounts, counts


def calculate_prior_terms(data: np.ndarray, noalle: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    nloci = data.shape[1]
    max_noalle = int(np.max(noalle, initial=0))
    adjprior = np.zeros((max_noalle, nloci), dtype=np.float64)
    prior_term = 0.0

    for locus_index in range(nloci):
        n_alleles = int(noalle[locus_index])
        if n_alleles == 0:
            continue
        adjprior[:n_alleles, locus_index] = alpha / n_alleles
        if n_alleles < max_noalle:
            adjprior[n_alleles:, locus_index] = 1.0
        prior_term += n_alleles * gammaln(alpha / n_alleles)

    return adjprior, prior_term


def create_baps_output_struct(
    partition: np.ndarray,
    snp_data: np.ndarray,
    snp_positions: np.ndarray,
    n_clusters: int,
    partition_array: np.ndarray | None = None,
    block_width: int | None = None,
    alpha: float = 1.0,
) -> BapsData:
    partition = np.asarray(partition, dtype=np.int64).reshape(-1)
    data, noalle = relabel_codes(snp_data)
    npops = int(np.unique(partition).size)

    if partition_array is None and block_width is None:
        sumcounts, counts = initial_counts(partition, data, npops, noalle)
    elif block_width is None:
        sumcounts, counts = initial_counts_with_pop_structure(
            np.asarray(partition_array, dtype=np.int64),
            data,
            noalle,
        )
    else:
        sumcounts, counts = initial_counts_with_origins(
            np.asarray(partition_array, dtype=np.int64),
            block_width,
            data,
            noalle,
            np.asarray(snp_positions, dtype=np.int64),
        )

    adjprior, prior_term = calculate_prior_terms(data, noalle, alpha)
    n_empty_clusters = n_clusters - counts.shape[2]
    if n_empty_clusters > 0:
        counts = np.concatenate(
            [
                counts,
                np.zeros((counts.shape[0], counts.shape[1], n_empty_clusters), dtype=counts.dtype),
            ],
            axis=2,
        )
        sumcounts = np.concatenate(
            [
                sumcounts,
                np.zeros((n_empty_clusters, sumcounts.shape[1]), dtype=sumcounts.dtype),
            ],
            axis=0,
        )

    return BapsData(
        PARTITION=partition,
        npops=npops,
        data=data,
        noalle=noalle,
        snpPositions=np.asarray(snp_positions, dtype=np.int64),
        SUMCOUNTS=sumcounts,
        COUNTS=counts,
        adjprior=adjprior,
        priorTerm=prior_term,
    )


def add_empty_cluster(c: BapsData) -> BapsData:
    n_snps = c.SUMCOUNTS.shape[1]
    max_noalle = c.COUNTS.shape[0]
    return replace(
        c,
        npops=c.npops + 1,
        SUMCOUNTS=np.concatenate([c.SUMCOUNTS, np.zeros((1, n_snps), dtype=c.SUMCOUNTS.dtype)], axis=0),
        COUNTS=np.concatenate(
            [c.COUNTS, np.zeros((max_noalle, n_snps, 1), dtype=c.COUNTS.dtype)],
            axis=2,
        ),
    )


def compute_diff_in_counts(rows: np.ndarray | list[int] | int, max_noalle: int, n_loci: int, data: np.ndarray) -> np.ndarray:
    if np.isscalar(rows):
        row_indices = np.array([int(rows)], dtype=np.int64)
    else:
        row_indices = np.asarray(rows, dtype=np.int64).reshape(-1)

    diff_in_counts = np.zeros((max_noalle, n_loci), dtype=np.float64)
    for row_index in row_indices:
        row = data[int(row_index) - 1, :].astype(np.int64, copy=False)
        not_empty = np.where(row > 0)[0]
        if not_empty.size == 0:
            continue
        diff_in_counts[row[not_empty] - 1, not_empty] += 1

    return diff_in_counts


def update_counts(c: BapsData, strain: int, new_states: np.ndarray, old_states: np.ndarray) -> BapsData:
    updated = replace(c, SUMCOUNTS=c.SUMCOUNTS.copy(), COUNTS=c.COUNTS.copy())
    strain_index = strain - 1
    changed_loci = np.where((new_states != old_states) & (updated.data[strain_index, :] > 0))[0]
    observations = updated.data[strain_index, changed_loci]

    for locus_index, observation in zip(changed_loci, observations, strict=True):
        old_cluster = int(old_states[locus_index]) - 1
        new_cluster = int(new_states[locus_index]) - 1
        allele_index = int(observation) - 1
        updated.COUNTS[allele_index, locus_index, old_cluster] -= 1
        updated.COUNTS[allele_index, locus_index, new_cluster] += 1
        updated.SUMCOUNTS[old_cluster, locus_index] -= 1
        updated.SUMCOUNTS[new_cluster, locus_index] += 1

    return updated


def update_counts_with_strain(
    c: BapsData,
    strain_index: int,
    new_states: np.ndarray | list[int],
    old_states: np.ndarray | list[int],
    change_only_these_sites: np.ndarray,
) -> BapsData:
    updated = replace(c, SUMCOUNTS=c.SUMCOUNTS.copy(), COUNTS=c.COUNTS.copy())
    diff_in_counts = compute_diff_in_counts(
        strain_index,
        updated.COUNTS.shape[0],
        updated.COUNTS.shape[1],
        updated.data,
    )
    diff_in_sumcounts = np.sum(diff_in_counts, axis=0)

    if len(old_states) == 0 or len(new_states) == 0:
        site_changed = np.ones(updated.COUNTS.shape[1], dtype=bool)
    else:
        site_changed = np.asarray(old_states) != np.asarray(new_states)
    site_changed = site_changed & np.asarray(change_only_these_sites, dtype=bool)

    if len(old_states) != 0:
        old_states_array = np.asarray(old_states, dtype=np.int64)
        losing_clusters = np.unique(old_states_array[site_changed])
        for losing_index in losing_clusters:
            losing_sites = (old_states_array == losing_index) & site_changed
            updated.COUNTS[:, losing_sites, losing_index - 1] -= diff_in_counts[:, losing_sites]
            updated.SUMCOUNTS[losing_index - 1, losing_sites] -= diff_in_sumcounts[losing_sites]

    if len(new_states) != 0:
        new_states_array = np.asarray(new_states, dtype=np.int64)
        gaining_clusters = np.unique(new_states_array[site_changed])
        for gaining_index in gaining_clusters:
            gaining_sites = (new_states_array == gaining_index) & site_changed
            updated.COUNTS[:, gaining_sites, gaining_index - 1] += diff_in_counts[:, gaining_sites]
            updated.SUMCOUNTS[gaining_index - 1, gaining_sites] += diff_in_sumcounts[gaining_sites]

    return updated
