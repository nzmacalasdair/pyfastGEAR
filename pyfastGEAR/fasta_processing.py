from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import PreprocessingResult

_BASE_TO_CODE = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "a": 1,
    "c": 2,
    "g": 3,
    "t": 4,
}
_MISSING_BASES = {"-", "?", "N", "n"}


def read_fasta_data(fasta_input_file: Path) -> PreprocessingResult:
    strain_labels, sequences = _read_fasta_records(fasta_input_file)
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_input_file}")

    total_sequence_length = len(sequences[0])
    for index, sequence in enumerate(sequences, start=1):
        if len(sequence) != total_sequence_length:
            raise ValueError(f"Sequence {index} has inconsistent length")

    observed_values = np.zeros((5, total_sequence_length), dtype=bool)
    for sequence in sequences:
        encoded = _encode_full_sequence(sequence)
        observed_values[encoded - 1, np.arange(total_sequence_length)] = True

    snp_positions = np.flatnonzero(np.sum(observed_values[:4, :], axis=0) > 1) + 1
    snp_data = np.zeros((len(sequences), int(snp_positions.size)), dtype=np.int16)

    snp_zero_based = snp_positions - 1
    for strain_index, sequence in enumerate(sequences):
        snp_data[strain_index, :] = _encode_snp_sequence(sequence, snp_zero_based)

    return PreprocessingResult(
        strain_labels=strain_labels,
        snp_data=snp_data,
        snp_positions=snp_positions.astype(np.int64),
        total_sequence_length=total_sequence_length,
    )


def _read_fasta_records(fasta_input_file: Path) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    sequences: list[str] = []

    current_label: str | None = None
    current_sequence: list[str] = []
    with fasta_input_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_label is not None:
                    labels.append(current_label)
                    sequences.append("".join(current_sequence))
                current_label = line[1:].strip()
                current_sequence = []
            else:
                current_sequence.append(line)

    if current_label is not None:
        labels.append(current_label)
        sequences.append("".join(current_sequence))

    return labels, sequences


def _encode_full_sequence(sequence: str) -> np.ndarray:
    encoded = np.full(len(sequence), 5, dtype=np.int16)
    for index, base in enumerate(sequence):
        if base in _BASE_TO_CODE:
            encoded[index] = _BASE_TO_CODE[base]
        elif base in _MISSING_BASES:
            encoded[index] = 5
        else:
            encoded[index] = 5
    return encoded


def _encode_snp_sequence(sequence: str, snp_zero_based: np.ndarray) -> np.ndarray:
    encoded = np.zeros(int(snp_zero_based.size), dtype=np.int16)
    for output_index, source_index in enumerate(snp_zero_based):
        base = sequence[int(source_index)]
        if base in _BASE_TO_CODE:
            encoded[output_index] = _BASE_TO_CODE[base]
        elif base in _MISSING_BASES:
            encoded[output_index] = -9
        else:
            encoded[output_index] = -9
    return encoded
