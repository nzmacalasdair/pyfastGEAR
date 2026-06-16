from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.special import gammaln

from .models import BapsData


def compute_locuswise_population_logml(
    pop_counts: np.ndarray,
    pop_sum_counts: np.ndarray,
    adjusted_prior_counts: np.ndarray,
    prior_sum_counts: np.ndarray,
) -> np.ndarray:
    return (
        gammaln(prior_sum_counts)
        - gammaln(pop_sum_counts + prior_sum_counts)
        + np.sum(gammaln(pop_counts + adjusted_prior_counts), axis=0)
        - np.sum(gammaln(adjusted_prior_counts), axis=0)
    )


def calc_emission_probs(
    counts: np.ndarray,
    sum_counts: np.ndarray,
    prior_counts: np.ndarray,
    prior_sum_counts: np.ndarray,
    data_row: np.ndarray,
) -> np.ndarray:
    max_noalle, nloci, npops = counts.shape
    emission_probs = np.zeros((npops, nloci), dtype=np.float64)

    total_counts = counts + prior_counts
    total_sum_counts = sum_counts + prior_sum_counts
    total_sum_counts = np.tile(total_sum_counts.T.reshape(1, nloci, npops), (max_noalle, 1, 1))
    all_freqs = total_counts / total_sum_counts

    non_missing = np.where(data_row > 0)[0]
    missing = np.setdiff1d(np.arange(nloci), non_missing)
    for pop_index in range(npops):
        if non_missing.size:
            allele_indices = data_row[non_missing].astype(np.int64) - 1
            emission_probs[pop_index, non_missing] = all_freqs[allele_indices, non_missing, pop_index]
        if missing.size:
            emission_probs[pop_index, missing] = 1.0

    return emission_probs


def calc_emission_probs_for_cluster_merging(
    counts: np.ndarray,
    sum_counts: np.ndarray,
    adjusted_prior_counts: np.ndarray,
    prior_sum_counts: np.ndarray,
    cluster_counts: np.ndarray,
    cluster_sum_counts: np.ndarray,
) -> np.ndarray:
    _, n_loci, npops = counts.shape
    emission_probs = np.zeros((npops, n_loci), dtype=np.float64)
    for pop_index in range(npops):
        numerator = compute_locuswise_population_logml(
            counts[:, :, pop_index] + cluster_counts,
            sum_counts[pop_index, :] + cluster_sum_counts,
            adjusted_prior_counts,
            prior_sum_counts,
        )
        denominator = compute_locuswise_population_logml(
            counts[:, :, pop_index],
            sum_counts[pop_index, :],
            adjusted_prior_counts,
            prior_sum_counts,
        )
        emission_probs[pop_index, :] = np.exp(numerator - denominator)
    return emission_probs


def calc_emission_probs_for_merging_two_clusters(
    counts1: np.ndarray,
    sum_counts1: np.ndarray,
    counts2: np.ndarray,
    sum_counts2: np.ndarray,
    adjusted_prior_counts: np.ndarray,
    prior_sum_counts: np.ndarray,
) -> np.ndarray:
    n_loci = counts1.shape[1]
    emission_probs = np.zeros((2, n_loci), dtype=np.float64)

    log_prob1 = compute_locuswise_population_logml(counts1, sum_counts1, adjusted_prior_counts, prior_sum_counts)
    log_prob2 = compute_locuswise_population_logml(counts2, sum_counts2, adjusted_prior_counts, prior_sum_counts)
    log_prob_separate = log_prob1 + log_prob2
    log_prob_merged = compute_locuswise_population_logml(
        counts1 + counts2,
        sum_counts1 + sum_counts2,
        adjusted_prior_counts,
        prior_sum_counts,
    )

    emission_probs[0, :] = np.exp(log_prob_merged)
    emission_probs[1, :] = np.exp(log_prob_separate)

    if np.any(emission_probs == 0):
        log_prob_both = np.vstack([log_prob_merged, log_prob_separate])
        smaller_log_prob = np.min(log_prob_both, axis=0)
        amount_too_small = np.maximum(-700.0 - smaller_log_prob, 0.0)
        log_prob_both = np.minimum(log_prob_both + amount_too_small, 0.0)
        emission_probs = np.exp(log_prob_both)

    return emission_probs


def pre_calculate_trans_matrices(
    total_sequence_length: int,
    trans_pars: float | Mapping[str, Any],
    n_states: int,
    snp_positions: np.ndarray,
    num_smallest_to_prevent: int = 0,
) -> dict[int, np.ndarray]:
    if not isinstance(trans_pars, Mapping):
        phi = float(trans_pars)
        rho = phi ** (1.0 / (total_sequence_length - 1))
        n_possible_states = n_states - num_smallest_to_prevent
        transition_probs_single_step = np.zeros((n_states, n_states), dtype=np.float64)
        start = num_smallest_to_prevent
        transition_probs_single_step[start:, start:] = (
            np.eye(n_states - start) * rho
            + (1.0 - np.eye(n_states - start)) * (1.0 - rho) / (n_states - 1 - start)
        )
        if num_smallest_to_prevent > 0:
            transition_probs_single_step[:start, start:] = 1.0 / n_possible_states
    else:
        home = int(trans_pars["home"])
        rho = float(trans_pars["rho"])
        rho0 = float(trans_pars["rho0"])
        a = float(trans_pars["a"])
        transition_probs_single_step = np.zeros((n_states, n_states), dtype=np.float64)
        for state_index in range(1, n_states + 1):
            if state_index == home:
                transition_probs_single_step[state_index - 1, :] = (1.0 - rho0) / (n_states - 1)
                transition_probs_single_step[state_index - 1, state_index - 1] = rho0
            else:
                transition_probs_single_step[state_index - 1, :] = (1.0 - rho) * (1.0 - a) / (n_states - 2)
                transition_probs_single_step[state_index - 1, home - 1] = (1.0 - rho) * a
                transition_probs_single_step[state_index - 1, state_index - 1] = rho

    snp_positions = np.asarray(snp_positions, dtype=np.int64)
    distance_list = np.unique(
        np.concatenate(
            [
                np.array([snp_positions[0] - 1], dtype=np.int64),
                np.diff(snp_positions),
                np.array([total_sequence_length - snp_positions[-1]], dtype=np.int64),
            ]
        )
    )
    distance_list = distance_list[distance_list != 0]
    return {int(dist): np.linalg.matrix_power(transition_probs_single_step, int(dist)) for dist in distance_list}


def calc_transition_probs(
    snp_distance: int,
    pre_calculated_trans_matrices: Mapping[int, np.ndarray] | list[np.ndarray],
) -> np.ndarray:
    if isinstance(pre_calculated_trans_matrices, Mapping):
        return pre_calculated_trans_matrices[int(snp_distance)]
    return pre_calculated_trans_matrices[int(snp_distance)]


def hmm_alpha_hat_recursion(
    snp_positions: np.ndarray,
    pre_calculated_trans_matrices: Mapping[int, np.ndarray] | list[np.ndarray],
    emission_probs: np.ndarray,
    init_transition_matrix: np.ndarray,
    init_alpha_hat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_states, n_snps = emission_probs.shape
    alpha_hats = np.zeros((n_states, n_snps), dtype=np.float64)
    coefs = np.zeros(n_snps, dtype=np.float64)

    if init_alpha_hat is None or np.asarray(init_alpha_hat).size == 0:
        init_alpha_hat = np.ones(n_states, dtype=np.float64) / n_states

    alpha_hats[:, 0] = emission_probs[:, 0] * (np.asarray(init_alpha_hat) @ init_transition_matrix)
    coefs[0] = np.sum(alpha_hats[:, 0])
    alpha_hats[:, 0] /= coefs[0]

    snp_distances = np.diff(snp_positions)
    for index in range(1, n_snps):
        transition_matrix = calc_transition_probs(int(snp_distances[index - 1]), pre_calculated_trans_matrices)
        alpha_hats[:, index] = (alpha_hats[:, index - 1] @ transition_matrix) * emission_probs[:, index]
        coefs[index] = np.sum(alpha_hats[:, index])
        alpha_hats[:, index] /= coefs[index]

    return alpha_hats, coefs


def hmm_beta_hat_recursion(
    coefs: np.ndarray,
    snp_positions: np.ndarray,
    pre_calculated_trans_matrices: Mapping[int, np.ndarray] | list[np.ndarray],
    emission_probs: np.ndarray,
    init_transition_matrix: np.ndarray | None = None,
    init_beta_hat: np.ndarray | None = None,
) -> np.ndarray:
    n_states, n_snps_plus_one = emission_probs.shape
    n_snps = n_snps_plus_one - 1
    beta_hats = np.zeros((n_states, n_snps), dtype=np.float64)

    if init_beta_hat is None or np.asarray(init_beta_hat).size == 0:
        beta_hats[:, n_snps - 1] = 1.0
    else:
        beta_hats[:, n_snps - 1] = init_transition_matrix @ (np.asarray(init_beta_hat) * emission_probs[:, n_snps])
        beta_hats[:, n_snps - 1] /= coefs[n_snps]

    snp_distances = np.diff(snp_positions)
    for index in range(n_snps - 2, -1, -1):
        transition_matrix = calc_transition_probs(int(snp_distances[index]), pre_calculated_trans_matrices)
        beta_hats[:, index] = transition_matrix @ (beta_hats[:, index + 1] * emission_probs[:, index + 1])
        beta_hats[:, index] /= coefs[index + 1]

    return beta_hats


def simulate_realization(
    marginals: np.ndarray,
    complete_beta_hat_list: np.ndarray,
    emission_probs: np.ndarray,
    snp_positions: np.ndarray,
    pre_calculated_trans_matrices: Mapping[int, np.ndarray] | list[np.ndarray],
    rng: np.random.Generator | None = None,
    first_uniform: float | None = None,
    transition_uniforms: np.ndarray | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    _, n_snps = marginals.shape
    states = np.zeros(n_snps, dtype=np.int64)
    first_draw = float(rng.random()) if first_uniform is None else float(first_uniform)
    states[0] = np.searchsorted(np.cumsum(marginals[:, 0]), first_draw, side="right") + 1

    snp_distances = np.diff(snp_positions)
    rand_numbers = rng.random(n_snps) if transition_uniforms is None else np.asarray(transition_uniforms, dtype=np.float64).reshape(-1)
    for index in range(1, n_snps):
        transition_matrix = calc_transition_probs(int(snp_distances[index - 1]), pre_calculated_trans_matrices)
        trans_dist = transition_matrix[states[index - 1] - 1, :]
        cond_dist = emission_probs[:, index] * trans_dist * complete_beta_hat_list[:, index]
        cond_dist /= np.sum(cond_dist)
        states[index] = np.searchsorted(np.cumsum(cond_dist), rand_numbers[index], side="right") + 1

    return states


def hmm_recombination_analysis(
    c: BapsData,
    pre_calculated_trans_matrices: Mapping[int, np.ndarray] | list[np.ndarray],
    strain_index: int | None = None,
    cluster_indices: np.ndarray | list[int] | None = None,
    rng: np.random.Generator | None = None,
    sampling_uniforms: tuple[float, np.ndarray] | None = None,
    return_debug: bool = False,
) -> tuple[np.ndarray, float, np.ndarray] | tuple[np.ndarray, float, np.ndarray, dict[str, np.ndarray]]:
    if np.any(c.COUNTS < 0):
        raise ValueError("Negative counts detected")

    if strain_index is not None:
        counts = c.COUNTS
        sum_counts = c.SUMCOUNTS
        prior_counts = c.adjprior.copy()
        prior_counts[prior_counts == 1] = 0
        prior_sum_counts = np.ones((1, len(c.noalle)), dtype=np.float64)

        n_clusters = c.COUNTS.shape[2]
        prior_counts = np.repeat(prior_counts[:, :, np.newaxis], n_clusters, axis=2)
        prior_sum_counts = np.tile(prior_sum_counts, (n_clusters, 1))

        home_cluster = int(c.PARTITION[strain_index - 1])
        home_cluster_size_vector = sum_counts[home_cluster - 1, :]
        for cluster_index in range(1, n_clusters):
            if cluster_index != home_cluster:
                cluster_size_vector = sum_counts[cluster_index - 1, :]
                loci_for_dispersion = np.where(
                    (home_cluster_size_vector > 0) & (cluster_size_vector > home_cluster_size_vector)
                )[0]
                if loci_for_dispersion.size > 0:
                    over_dispersions = 1.1 * cluster_size_vector[loci_for_dispersion] / home_cluster_size_vector[loci_for_dispersion]
                    prior_counts[:, loci_for_dispersion, cluster_index - 1] *= over_dispersions
                    prior_sum_counts[cluster_index - 1, loci_for_dispersion] *= over_dispersions

        emission_probs = calc_emission_probs(
            counts,
            sum_counts,
            prior_counts,
            prior_sum_counts,
            c.data[strain_index - 1, :],
        )
        n_states = c.COUNTS.shape[2]
    else:
        assert cluster_indices is not None
        cluster_indices = np.asarray(cluster_indices, dtype=np.int64)
        counts1 = c.COUNTS[:, :, cluster_indices[0] - 1]
        counts2 = c.COUNTS[:, :, cluster_indices[1] - 1]
        sum_counts1 = c.SUMCOUNTS[cluster_indices[0] - 1, :]
        sum_counts2 = c.SUMCOUNTS[cluster_indices[1] - 1, :]
        adjusted_prior_counts = c.adjprior.copy()
        adjusted_prior_counts[adjusted_prior_counts == 1] = np.nan
        adjusted_prior_counts[np.isnan(adjusted_prior_counts)] = 1.0
        prior_sum_counts = np.ones((1, len(c.noalle)), dtype=np.float64)
        emission_probs = calc_emission_probs_for_merging_two_clusters(
            counts1,
            sum_counts1,
            counts2,
            sum_counts2,
            adjusted_prior_counts,
            prior_sum_counts,
        )
        n_states = 2

    complete_alpha_hat_list = np.zeros((n_states, c.data.shape[1]), dtype=np.float64)
    complete_coef_list = np.zeros(c.data.shape[1], dtype=np.float64)
    init_transition_matrix = np.ones((n_states, n_states), dtype=np.float64) / n_states
    alpha_hats, coefs = hmm_alpha_hat_recursion(
        c.snpPositions,
        pre_calculated_trans_matrices,
        emission_probs,
        init_transition_matrix,
        None,
    )
    n_positions = len(c.snpPositions)
    complete_alpha_hat_list[:, :n_positions] = alpha_hats
    complete_coef_list[:n_positions] = coefs

    complete_beta_hat_list = np.zeros((n_states, c.data.shape[1]), dtype=np.float64)
    emission_probs_with_terminal = np.column_stack([emission_probs, np.ones(n_states, dtype=np.float64)])
    coefs_extended = np.concatenate([complete_coef_list[:n_positions], [1.0]])
    beta_hats = hmm_beta_hat_recursion(
        coefs_extended,
        c.snpPositions,
        pre_calculated_trans_matrices,
        emission_probs_with_terminal,
        None,
        None,
    )
    complete_beta_hat_list[:, :n_positions] = beta_hats

    marginals = complete_alpha_hat_list[:, :n_positions] * complete_beta_hat_list[:, :n_positions]
    log_posterior = float(np.sum(np.log(complete_coef_list[:n_positions])))
    states = simulate_realization(
        marginals,
        complete_beta_hat_list[:, :n_positions],
        emission_probs_with_terminal,
        c.snpPositions,
        pre_calculated_trans_matrices,
        rng=rng,
        first_uniform=None if sampling_uniforms is None else sampling_uniforms[0],
        transition_uniforms=None if sampling_uniforms is None else sampling_uniforms[1],
    )
    if return_debug:
        return marginals, log_posterior, states, {
            "complete_beta_hat_list": complete_beta_hat_list[:, :n_positions],
            "emission_probs_with_terminal": emission_probs_with_terminal,
        }
    return marginals, log_posterior, states
