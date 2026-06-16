from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pickle
import numpy as np

from .clustering import learn_clustering
from .fasta_processing import read_fasta_data
from .generate_output import ensure_output_dir, save_pickle, write_outputs
from .models import FastGearConfig, UserData
from .recombination import learn_recombinations_in_clusters, learn_recombinations_in_clusters_one_lineage
from .relationships import run_post_recombination_relationships, run_pre_recombination_relationships
from .significance import (
    clear_non_significant_ancestral_recombinations,
    clear_non_significant_recent_recombinations,
)


def initialize_user_data(config: FastGearConfig) -> UserData:
    preprocessing = read_fasta_data(config.input_file)
    snp_data_file = _snp_data_path(config.output_file)

    user_data = UserData(
        n_iter=config.n_iterations,
        n_iter_completed=0,
        strain_labels=preprocessing.strain_labels,
        snp_positions=preprocessing.snp_positions,
        total_sequence_length=preprocessing.total_sequence_length,
        snp_data_file_name=str(snp_data_file),
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

    save_pickle(snp_data_file, {"snpData": preprocessing.snp_data})
    save_pickle(config.output_file, {"uData": user_data.to_dict()})
    return user_data


def run_preprocessing_only(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    return initialize_user_data(config)


def run_preprocessing_and_clustering(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data = initialize_user_data(config)
    with Path(user_data.snp_data_file_name).open("rb") as handle:
        snp_payload = pickle.load(handle)
    snp_data = np.asarray(snp_payload["snpData"])
    user_data = learn_clustering(
        user_data,
        snp_data,
        config.n_clust_list,
        config.run_all_upper_bounds,
        np.random.RandomState(1),
    )
    save_pickle(config.output_file, {"uData": user_data.to_dict()})
    return user_data


def run_preprocessing_clustering_and_relationships(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data = initialize_user_data(config)
    with Path(user_data.snp_data_file_name).open("rb") as handle:
        snp_payload = pickle.load(handle)
    snp_data = np.asarray(snp_payload["snpData"])
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
    )
    save_pickle(config.output_file, {"uData": user_data.to_dict()})
    return user_data


def run_to_one_lineage_recombination(config: FastGearConfig) -> UserData:
    if config.run_recombination_lineage is None:
        raise ValueError("run_recombination_lineage must be set")
    ensure_output_dir(config)
    user_data = initialize_user_data(config)
    with Path(user_data.snp_data_file_name).open("rb") as handle:
        snp_payload = pickle.load(handle)
    snp_data = np.asarray(snp_payload["snpData"])
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
    )
    user_data, recomb_result = learn_recombinations_in_clusters_one_lineage(
        user_data,
        snp_data,
        config.run_recombination_lineage,
        config.resolved_output_dir(),
        config.reduced_output,
        rng=np.random.default_rng(1),
    )
    save_pickle(config.resolved_output_dir() / f"lineage_{config.run_recombination_lineage}_recombination.pkl", recomb_result)
    save_pickle(config.output_file, {"uData": user_data.to_dict()})
    return user_data


def run_to_recent_recombination(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data = initialize_user_data(config)
    with Path(user_data.snp_data_file_name).open("rb") as handle:
        snp_payload = pickle.load(handle)
    snp_data = np.asarray(snp_payload["snpData"])
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
    save_pickle(config.output_file, {"uData": user_data.to_dict()})
    return user_data


def run_to_significance(config: FastGearConfig) -> UserData:
    ensure_output_dir(config)
    user_data = initialize_user_data(config)
    with Path(user_data.snp_data_file_name).open("rb") as handle:
        snp_payload = pickle.load(handle)
    snp_data = np.asarray(snp_payload["snpData"])
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
    )
    user_data = clear_non_significant_ancestral_recombinations(user_data, snp_data, 1)
    user_data = clear_non_significant_ancestral_recombinations(user_data, snp_data, 2)
    user_data = clear_non_significant_recent_recombinations(user_data, snp_data)
    user_data = replace(user_data, state="significancesComputed")
    save_pickle(config.output_file, {"uData": user_data.to_dict()})
    return user_data


def run_to_outputs(config: FastGearConfig) -> UserData:
    user_data = run_to_significance(config)
    write_outputs(user_data, config.resolved_output_dir())
    save_pickle(config.output_file, {"uData": user_data.to_dict()})
    return user_data


def _snp_data_path(output_file: Path) -> Path:
    suffix = output_file.suffix or ".pkl"
    return output_file.with_name(f"{output_file.stem}_snpData{suffix}")
