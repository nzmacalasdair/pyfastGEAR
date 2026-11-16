from __future__ import annotations

import os
from dataclasses import replace
import warnings

import numpy as np

from .clustering import learn_clustering
from .fasta_processing import read_fasta_data
from .generate_output import ensure_output_dir, save_pickle, write_outputs
from .handle_input import has_advanced_stage_selection
from .models import FastGearConfig, UserData
from .recombination import learn_recombinations_in_clusters, learn_recombinations_in_clusters_one_lineage
from .relationships import run_post_recombination_relationships, run_pre_recombination_relationships
from .significance import (
    clear_non_significant_ancestral_recombinations,
    clear_non_significant_recent_recombinations,
)


def initialize_user_data(config: FastGearConfig) -> tuple[UserData, np.ndarray]:
    preprocessing = read_fasta_data(config.input_file)
    snp_data_file = config.resolved_output_dir() / "snp_data.pkl"

    user_data = UserData(
        n_iter=config.n_iterations,
        n_iter_completed=0,
        strain_labels=preprocessing.strain_labels,
        snp_positions=preprocessing.snp_positions,
        total_sequence_length=preprocessing.total_sequence_length,
        snp_data_file_name=str(snp_data_file) if not config.reduced_output else "",
        output_dir=str(config.resolved_output_dir()),
        delimiter=os.sep,
        state="dataLoaded",
        trans_pars={
            "muMax": 20000.0,
            "muMin": 10.0,
            "rho": (10005.0 - 1.0) / 10005.0,
            "rho0": 0.5 ** (1.0 / preprocessing.total_sequence_length),
            "aPriorCount": 10.0,
            "negAPriorCount": 1.5,
            "a": 10.0 / 11.5,
        },
    )

    if not config.reduced_output:
        save_pickle(snp_data_file, {"snpData": preprocessing.snp_data})
    _save_run_state(config, user_data)
    return user_data, preprocessing.snp_data


def run_preprocessing_only(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data, snp_data = initialize_user_data(config)
    if _has_insufficient_snps(user_data):
        return _complete_insufficient_snp_analysis(config, user_data)
    return user_data


def run_preprocessing_and_clustering(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data, snp_data = initialize_user_data(config)
    if _has_insufficient_snps(user_data):
        return _complete_insufficient_snp_analysis(config, user_data)
    user_data = learn_clustering(
        user_data,
        snp_data,
        config.n_clust_list,
        config.run_all_upper_bounds,
        np.random.RandomState(1),
    )
    _save_run_state(config, user_data)
    return user_data


def run_preprocessing_clustering_and_relationships(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data, snp_data = initialize_user_data(config)
    if _has_insufficient_snps(user_data):
        return _complete_insufficient_snp_analysis(config, user_data)
    user_data = learn_clustering(
        user_data,
        snp_data,
        config.n_clust_list,
        config.run_all_upper_bounds,
        np.random.RandomState(1),
    )
    user_data, _ = run_pre_recombination_relationships(
        user_data,
        snp_data,
        config.resolved_output_dir(),
        reduced_output=config.reduced_output,
    )
    _save_run_state(config, user_data)
    return user_data


def run_to_one_lineage_recombination(config: FastGearConfig) -> UserData:
    if config.run_recombination_lineage is None:
        raise ValueError("run_recombination_lineage must be set")
    ensure_output_dir(config)
    user_data, snp_data = initialize_user_data(config)
    if _has_insufficient_snps(user_data):
        return _complete_insufficient_snp_analysis(config, user_data)
    user_data = learn_clustering(
        user_data,
        snp_data,
        config.n_clust_list,
        config.run_all_upper_bounds,
        np.random.RandomState(1),
    )
    user_data, _ = run_pre_recombination_relationships(
        user_data,
        snp_data,
        config.resolved_output_dir(),
        reduced_output=config.reduced_output,
    )
    user_data, recomb_result = learn_recombinations_in_clusters_one_lineage(
        user_data,
        snp_data,
        config.run_recombination_lineage,
        config.resolved_output_dir(),
        config.reduced_output,
        rng=np.random.default_rng(1),
    )
    if not config.reduced_output:
        save_pickle(config.resolved_output_dir() / f"lineage_{config.run_recombination_lineage}_recombination.pkl", recomb_result)
    _save_run_state(config, user_data)
    return user_data


def run_to_recent_recombination(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data, snp_data = initialize_user_data(config)
    if _has_insufficient_snps(user_data):
        return _complete_insufficient_snp_analysis(config, user_data)
    user_data = learn_clustering(
        user_data,
        snp_data,
        config.n_clust_list,
        config.run_all_upper_bounds,
        np.random.RandomState(1),
    )
    user_data, _ = run_pre_recombination_relationships(
        user_data,
        snp_data,
        config.resolved_output_dir(),
        reduced_output=config.reduced_output,
    )
    user_data, lineage_results = learn_recombinations_in_clusters(
        user_data,
        snp_data,
        config.resolved_output_dir(),
        config.reduced_output,
        rng=np.random.default_rng(1),
    )
    if not config.reduced_output:
        save_pickle(config.resolved_output_dir() / "recent_recombination.pkl", {"lineageResults": lineage_results})
    _save_run_state(config, user_data)
    return user_data


def run_to_significance(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data, snp_data = initialize_user_data(config)
    if _has_insufficient_snps(user_data):
        return _complete_insufficient_snp_analysis(config, user_data)
    user_data = learn_clustering(
        user_data,
        snp_data,
        config.n_clust_list,
        config.run_all_upper_bounds,
        np.random.RandomState(1),
    )
    user_data, _ = run_pre_recombination_relationships(
        user_data,
        snp_data,
        config.resolved_output_dir(),
        reduced_output=config.reduced_output,
    )
    user_data, _ = learn_recombinations_in_clusters(
        user_data,
        snp_data,
        config.resolved_output_dir(),
        config.reduced_output,
        rng=np.random.default_rng(1),
    )
    user_data, _ = run_post_recombination_relationships(
        user_data,
        snp_data,
        config.resolved_output_dir(),
        reduced_output=config.reduced_output,
    )
    user_data = clear_non_significant_ancestral_recombinations(user_data, snp_data, 1)
    user_data = clear_non_significant_ancestral_recombinations(user_data, snp_data, 2)
    user_data = clear_non_significant_recent_recombinations(user_data, snp_data)
    user_data = replace(user_data, state="significancesComputed")
    _save_run_state(config, user_data)
    return user_data


def run_to_outputs(config: FastGearConfig) -> UserData:
    user_data = run_to_significance(config)
    write_outputs(user_data, config.resolved_output_dir())
    _save_run_state(config, user_data)
    return user_data


def _save_run_state(config: FastGearConfig, user_data: UserData) -> None:
    if not config.reduced_output:
        save_pickle(config.resolved_output_dir() / "run_state.pkl", {"uData": user_data.to_dict()})


def _has_insufficient_snps(user_data: UserData) -> bool:
    return len(user_data.snp_positions) <= 1


def _complete_insufficient_snp_analysis(config: FastGearConfig, user_data: UserData) -> UserData:
    n_strains = len(user_data.strain_labels)
    n_snps = len(user_data.snp_positions)
    warnings.warn(
        f"Found {n_snps} SNP(s); skipping fastGEAR analysis because at least two "
        "SNPs are required.",
        UserWarning,
        stacklevel=2,
    )

    one_lineage = np.ones(n_strains, dtype=np.int64)
    one_structure = np.ones((1, n_snps), dtype=np.uint8)
    completed = replace(
        user_data,
        n_iter_completed=user_data.n_iter,
        state="analysisSkippedInsufficientSnps",
        partition=one_lineage.copy(),
        grouped_partition=one_lineage.copy(),
        cluster_dist=np.zeros((1, 1), dtype=np.float64),
        cluster_groups=np.ones(1, dtype=np.int64),
        cluster_structure=one_structure.copy(),
        lineage_structure=one_structure.copy(),
        lineage_structure_cleaned=one_structure.copy(),
        cluster_structure_cleaned=one_structure.copy(),
        updated_pop_structure=np.tile(one_structure, (n_strains, 1)),
        pop_structure_cleaned=np.tile(one_structure, (n_strains, 1)),
        recent_recombination_significances=[np.zeros(0, dtype=np.float64) for _ in range(n_strains)],
        lineage_ancestral_log_bf=[np.zeros(0, dtype=np.float64)],
        cluster_ancestral_log_bf=[np.zeros(0, dtype=np.float64)],
        output_dir=str(config.resolved_output_dir()),
    )
    _save_run_state(config, completed)
    if config.write_outputs or not has_advanced_stage_selection(config):
        write_outputs(completed, config.resolved_output_dir())
        _save_run_state(config, completed)
    return completed
