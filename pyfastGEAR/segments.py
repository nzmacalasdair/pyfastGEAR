from __future__ import annotations

import numpy as np


def identify_segments(
    pop_structure: np.ndarray,
    strain: int,
    snp_positions: np.ndarray,
    total_sequence_length: int,
) -> np.ndarray:
    if len(snp_positions) < 2:
        return np.zeros((0, 3), dtype=np.int64)

    pop_structure_strain = np.asarray(pop_structure[strain - 1, :], dtype=np.int64)
    break_points = np.where(pop_structure_strain[1:] - pop_structure_strain[:-1] != 0)[0] + 1
    segments = np.zeros((10, 3), dtype=np.int64)
    n_segments = 0

    for i, break_point in enumerate(break_points, start=1):
        n_segments += 1
        if n_segments > segments.shape[0]:
            segments = np.vstack([segments, np.zeros_like(segments)])
        if i == 1:
            starting = 1
            origin = int(pop_structure_strain[0])
        else:
            prev_break = break_points[i - 2]
            starting = int(np.floor((snp_positions[prev_break - 1] + snp_positions[prev_break]) / 2) + 1)
            origin = int(pop_structure_strain[break_point - 1])
        ending = int(np.floor((snp_positions[break_point - 1] + snp_positions[break_point]) / 2))
        segments[i - 1, :] = [starting, ending, origin]

    n_segments += 1
    if n_segments > segments.shape[0]:
        segments = np.vstack([segments, np.zeros_like(segments)])
    if len(break_points) == 0:
        segments[n_segments - 1, :] = [1, total_sequence_length, int(pop_structure_strain[0])]
    else:
        last_break = break_points[-1]
        segments[n_segments - 1, :] = [
            int(np.floor((snp_positions[last_break - 1] + snp_positions[last_break]) / 2) + 1),
            int(total_sequence_length),
            int(pop_structure_strain[-1]),
        ]
    return segments[:n_segments, :]
