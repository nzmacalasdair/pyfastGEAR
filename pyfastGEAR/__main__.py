from __future__ import annotations

from .generate_output import infer_output_dir
from .handle_input import config_from_args, has_advanced_stage_selection
from .pipeline import (
    run_preprocessing_and_clustering,
    run_preprocessing_clustering_and_relationships,
    run_to_significance,
    run_to_outputs,
    run_to_recent_recombination,
    run_to_one_lineage_recombination,
)


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(argv)
    if config.output_dir is None:
        config.output_dir = infer_output_dir(config.output_file)

    if has_advanced_stage_selection(config):
        if config.run_recombination_lineage is not None:
            user_data = run_to_one_lineage_recombination(config)
        elif config.write_outputs:
            user_data = run_to_outputs(config)
        elif config.run_significance:
            user_data = run_to_significance(config)
        elif config.run_recombination:
            user_data = run_to_recent_recombination(config)
        elif config.run_relationships:
            user_data = run_preprocessing_clustering_and_relationships(config)
        elif config.run_clustering:
            user_data = run_preprocessing_and_clustering(config)
        else:
            user_data = run_to_outputs(config)
    else:
        user_data = run_to_outputs(config)
    print(f"Loaded {len(user_data.strain_labels)} strains")
    print(f"Detected {len(user_data.snp_positions)} SNPs")
    if user_data.partition is not None:
        print(f"Detected {len(set(user_data.partition.tolist()))} clusters")
    if user_data.grouped_partition is not None:
        print(f"Detected {len(set(user_data.grouped_partition.tolist()))} lineages")
    if user_data.updated_pop_structure is not None:
        print(f"Computed recent recombination profiles with shape {tuple(user_data.updated_pop_structure.shape)}")
    print(f"Output directory: {config.resolved_output_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
