from __future__ import annotations

import argparse
from argparse import RawDescriptionHelpFormatter
from pathlib import Path

from .models import FastGearConfig


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pyfastGEAR on a FASTA alignment and produce final lineage/recombination outputs.",
        epilog=(
            "Examples:\n"
            "  python -m pyfastGEAR alignment.fa results.pkl\n"
            "  python -m pyfastGEAR alignment.fa results.pkl --output-dir results_output\n"
            "  python -m pyfastGEAR alignment.fa results.pkl --input-specs-file fG_input_specs.txt"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", type=Path, help="Input FASTA alignment")
    parser.add_argument("output_file", type=Path, help="Output run-state pickle")
    parser.add_argument(
        "--input-specs-file",
        type=Path,
        default=None,
        help="Optional MATLAB-compatible fG_input_specs.txt file",
    )
    parser.add_argument(
        "--partition-file",
        default="-",
        help="Use a predefined partition file instead of learning clusters; use '-' to infer clusters",
    )
    parser.add_argument("--n-iterations", type=int, default=15, help="Number of recent-recombination iterations")
    parser.add_argument(
        "--n-clust-list",
        type=int,
        nargs="+",
        default=[10, 15, 20, 30],
        help="Candidate upper bounds for clustering",
    )
    parser.add_argument(
        "--run-all-upper-bounds",
        action="store_true",
        help="Evaluate all configured clustering upper bounds instead of selecting one",
    )
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Write detailed intermediate artifacts in addition to the main outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for text outputs and optional artifacts",
    )

    advanced = parser.add_argument_group("advanced debugging")
    advanced.add_argument(
        "--run-clustering",
        action="store_true",
        help="Stop after stage-4 clustering",
    )
    advanced.add_argument(
        "--run-relationships",
        action="store_true",
        help="Stop after stage-5 pre-recombination relationships",
    )
    advanced.add_argument(
        "--run-recombination",
        action="store_true",
        help="Stop after stage-6 recent recombination",
    )
    advanced.add_argument(
        "--run-significance",
        action="store_true",
        help="Stop after stage-7 significance cleaning",
    )
    advanced.add_argument(
        "--write-outputs",
        action="store_true",
        help="Run the full pipeline through text output writing",
    )
    advanced.add_argument(
        "--run-recombination-lineage",
        type=int,
        default=None,
        help="Debug mode: run stage-6 one-lineage recent recombination for the given lineage index",
    )
    return parser


def parse_numeric_list(value: str) -> list[int]:
    cleaned = value.replace(",", " ").split()
    return [int(part) for part in cleaned]


def parse_input_specs_file(input_specs_file: Path) -> dict[str, object]:
    # MATLAB-era config files may contain odd non-UTF-8 bytes in comments.
    # Decode permissively because only the tokens before '#' are significant.
    lines = input_specs_file.read_bytes().decode("latin1").splitlines()
    if len(lines) < 5:
        raise ValueError(f"Expected at least 5 lines in {input_specs_file}")

    values = []
    for line in lines[:5]:
        comment_index = line.find("#")
        if comment_index != -1:
            line = line[:comment_index]
        values.append(line.strip())

    return {
        "n_iterations": int(values[0]),
        "n_clust_list": parse_numeric_list(values[1]),
        "run_all_upper_bounds": bool(int(values[2])),
        "partition_file": values[3].replace(" ", ""),
        "reduced_output": bool(int(values[4])),
    }


def config_from_args(argv: list[str] | None = None) -> FastGearConfig:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    config = FastGearConfig(
        input_file=args.input_file,
        output_file=args.output_file,
        input_specs_file=args.input_specs_file,
        partition_file=args.partition_file,
        n_iterations=args.n_iterations,
        n_clust_list=list(args.n_clust_list),
        run_all_upper_bounds=args.run_all_upper_bounds,
        reduced_output=not args.full_output,
        output_dir=args.output_dir,
        run_clustering=args.run_clustering,
        run_relationships=args.run_relationships,
        run_recombination=args.run_recombination,
        run_significance=args.run_significance,
        write_outputs=args.write_outputs,
        run_recombination_lineage=args.run_recombination_lineage,
    )

    if args.input_specs_file is not None:
        apply_legacy_specs(config, args.input_specs_file)

    return config


def apply_legacy_specs(config: FastGearConfig, input_specs_file: Path) -> None:
    specs = parse_input_specs_file(input_specs_file)
    config.input_specs_file = input_specs_file
    config.partition_file = str(specs["partition_file"])
    config.n_iterations = int(specs["n_iterations"])
    config.n_clust_list = list(specs["n_clust_list"])
    config.run_all_upper_bounds = bool(specs["run_all_upper_bounds"])
    config.reduced_output = bool(specs["reduced_output"])


def has_advanced_stage_selection(config: FastGearConfig) -> bool:
    return any(
        [
            config.run_clustering,
            config.run_relationships,
            config.run_recombination,
            config.run_significance,
            config.write_outputs,
            config.run_recombination_lineage is not None,
        ]
    )
