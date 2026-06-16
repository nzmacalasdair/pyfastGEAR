from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.optimize import fmin
from scipy.special import gammaln

from .counts import calculate_prior_terms, compute_diff_in_counts, relabel_codes
from .models import BapsData, UserData


LOGML_EPS = 1e-5
NEG_INF = -1e50


def get_baps_input_block_from_snp_data(
    block_index: int,
    block_width: int,
    snp_data: np.ndarray,
    snp_positions: np.ndarray,
    total_sequence_length: int,
    alpha: float = 1.0,
) -> BapsData:
    first_position = (block_index - 1) * block_width + 1
    if first_position > total_sequence_length:
        raise ValueError("Block index out of range.")
    last_position = min(block_index * block_width, total_sequence_length)
    columns_to_include = (snp_positions >= first_position) & (snp_positions <= last_position)

    data, noalle = relabel_codes(snp_data[:, columns_to_include])
    if np.any(noalle == 1):
        raise ValueError("SNP with only a single value.")
    adjprior, prior_term = calculate_prior_terms(data, noalle, alpha)
    return BapsData(
        PARTITION=np.zeros(data.shape[0], dtype=np.int64),
        npops=0,
        data=data,
        noalle=noalle,
        snpPositions=np.asarray(snp_positions[columns_to_include], dtype=np.int64),
        SUMCOUNTS=np.zeros((0, data.shape[1]), dtype=np.float64),
        COUNTS=np.zeros((int(np.max(noalle, initial=0)), data.shape[1], 0), dtype=np.float64),
        adjprior=adjprior,
        priorTerm=prior_term,
    )


def update_struct(c: BapsData) -> BapsData:
    ninds = c.data.shape[0]
    rows_from_ind = 1
    rows = np.column_stack(
        [
            np.arange(1, ninds + 1, dtype=np.int64),
            np.arange(1, ninds + 1, dtype=np.int64),
        ]
    )
    z, dist = new_get_distances(c.data, rows_from_ind)
    return replace(c, rowsFromInd=rows_from_ind, rows=rows, Z=z, dist=dist)


def learn_dir_hyper_par(
    counts: np.ndarray,
    noalle: np.ndarray,
    lambda_param: float | None = None,
    search_range: np.ndarray | None = None,
) -> np.ndarray | float:
    counts_first = counts[:, :, 0]
    n_snps = counts_first.shape[1]

    if lambda_param is None:
        alpha_array = np.zeros(n_snps, dtype=np.float64)
        for locus_index in range(n_snps):
            n = counts_first[: int(noalle[locus_index]), locus_index]
            if n.size == 0:
                continue

            def objective(a: np.ndarray) -> float:
                alpha = float(a[0])
                if alpha <= 0:
                    return np.inf
                term = -gammaln(alpha) + gammaln(alpha + np.sum(n))
                term -= np.sum(gammaln(alpha / n.size + n))
                term += n.size * gammaln(alpha / n.size)
                return -float(term)

            alpha_array[locus_index] = float(fmin(objective, np.array([1.0]), disp=False)[0])
        return alpha_array

    adjprior = np.zeros_like(counts_first, dtype=np.float64)
    for locus_index in range(n_snps):
        n_alleles = int(noalle[locus_index])
        if n_alleles > 0:
            adjprior[:n_alleles, locus_index] = 1.0 / n_alleles
    non_zero = adjprior > 0

    if search_range is None:
        def objective(a: np.ndarray) -> float:
            alpha = float(a[0])
            if alpha <= 0:
                return np.inf
            value = (
                lambda_param * alpha
                - n_snps * gammaln(alpha)
                + np.sum(gammaln(alpha + np.sum(counts_first, axis=0)))
                - np.sum(gammaln(alpha * adjprior[non_zero] + counts_first[non_zero]))
                + np.sum(gammaln(alpha * adjprior[non_zero]))
            )
            return float(value)

        return float(fmin(objective, np.array([1.0]), disp=False)[0])

    posterior = np.zeros(len(search_range), dtype=np.float64)
    for index, alpha in enumerate(search_range):
        posterior[index] = (
            -lambda_param * alpha
            + n_snps * gammaln(alpha)
            - np.sum(gammaln(alpha + np.sum(counts_first, axis=0)))
            + np.sum(gammaln(alpha * adjprior[non_zero] + counts_first[non_zero]))
            - np.sum(gammaln(alpha * adjprior[non_zero]))
        )
    posterior -= np.max(posterior)
    posterior = np.exp(posterior)
    posterior /= np.sum(posterior)
    return posterior


def do_baps_clusterings_in_blocks(
    snp_data: np.ndarray,
    snp_positions: np.ndarray,
    total_sequence_length: int,
    n_snps_lower_limit: int,
    block_width: int,
    alpha: float = 1.0,
    initial_num_pops: int | list[int] | np.ndarray | None = None,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.RandomState(1)
    if initial_num_pops is None:
        initial_num_pops_array: np.ndarray | None = None
    else:
        initial_num_pops_array = np.atleast_1d(np.asarray(initial_num_pops, dtype=np.int64))

    n_blocks = int(np.ceil(total_sequence_length / block_width))
    n_sequences = snp_data.shape[0]
    npops_array = np.zeros(n_blocks, dtype=np.int64)
    partition_array = np.zeros((n_sequences, n_blocks), dtype=np.int64)
    n_snps_array = np.zeros(n_blocks, dtype=np.int64)

    indices_of_too_small_blocks: list[int] = []
    c_old: BapsData | None = None

    for block_index in range(1, n_blocks + 1):
        c = get_baps_input_block_from_snp_data(
            block_index,
            block_width,
            snp_data,
            snp_positions,
            total_sequence_length,
            alpha,
        )
        n_snps_array[block_index - 1] = len(c.noalle)

        if c_old is not None:
            c = _concatenate_baps_blocks(c_old, c)

        if len(c.noalle) >= n_snps_lower_limit or block_index == n_blocks:
            empty_seqs = np.where(np.all(c.data == 0, axis=1))[0]
            non_empty_data = np.delete(c.data, empty_seqs, axis=0)

            if non_empty_data.size > 0:
                c_work = update_struct(replace(c, data=non_empty_data))
                if initial_num_pops_array is None:
                    partition, npops, _ = _run_default_ind_mix(c_work, alpha, rng)
                else:
                    logml_list = np.zeros(len(initial_num_pops_array), dtype=np.float64)
                    partition_list = np.zeros((c_work.data.shape[0], len(initial_num_pops_array)), dtype=np.int64)
                    for init_index, init_pops_now in enumerate(initial_num_pops_array):
                        logml, npops, _, partition_candidate = ind_mix(
                            c_work,
                            int(init_pops_now),
                            False,
                            alpha,
                            rng,
                        )
                        partition_list[:, init_index] = partition_candidate
                        logml_list[init_index] = logml
                    max_index = int(np.argmax(logml_list))
                    partition = partition_list[:, max_index]
                    npops = int(np.unique(partition).size)
            else:
                last_non_empty_index = min(indices_of_too_small_blocks + [block_index - 1]) - 1
                partition = partition_array[:, last_non_empty_index]
                npops = int(np.unique(partition).size)
                empty_seqs = np.array([], dtype=np.int64)

            if empty_seqs.size > 0:
                non_empty_seqs = np.setdiff1d(np.arange(n_sequences), empty_seqs)
                partition_new = np.zeros(n_sequences, dtype=np.int64)
                partition_new[non_empty_seqs] = partition
                partition_new[empty_seqs] = rng.randint(1, npops + 1, size=empty_seqs.size)
                partition = partition_new

            fill_columns = indices_of_too_small_blocks + [block_index - 1]
            partition_array[:, fill_columns] = partition[:, np.newaxis]
            npops_array[fill_columns] = npops
            c_old = None
            indices_of_too_small_blocks = []
        else:
            c_old = c
            indices_of_too_small_blocks.append(block_index - 1)

    return partition_array, n_snps_array, npops_array


def learn_clustering(
    user_data: UserData,
    snp_data: np.ndarray,
    n_clust_list: list[int],
    run_all_upper_bounds: bool,
    rng: np.random.RandomState | None = None,
) -> UserData:
    if rng is None:
        rng = np.random.RandomState(1)

    if run_all_upper_bounds:
        initial_num_pops: int | list[int] = n_clust_list
        partition_array, _, _ = do_baps_clusterings_in_blocks(
            snp_data,
            user_data.snp_positions,
            user_data.total_sequence_length,
            0,
            user_data.total_sequence_length,
            user_data.alpha,
            initial_num_pops,
            rng,
        )
    else:
        upper_bound_index = 0
        partition_array = np.zeros((snp_data.shape[0], 1), dtype=np.int64)
        while (
            upper_bound_index == 0
            or (
                len(np.unique(partition_array)) == n_clust_list[upper_bound_index - 1]
                and upper_bound_index < len(n_clust_list)
            )
        ):
            initial_num_pops_now = n_clust_list[upper_bound_index]
            partition_array, _, _ = do_baps_clusterings_in_blocks(
                snp_data,
                user_data.snp_positions,
                user_data.total_sequence_length,
                0,
                user_data.total_sequence_length,
                user_data.alpha,
                initial_num_pops_now,
                rng,
            )
            upper_bound_index += 1
            if upper_bound_index >= len(n_clust_list):
                break

    partition = reorder_cluster_labels(partition_array[:, 0])
    return replace(user_data, partition=partition, state="clusteringDone")


def reorder_cluster_labels(partition: np.ndarray) -> np.ndarray:
    partition = np.asarray(partition, dtype=np.int64).reshape(-1)
    unique_clusters = np.unique(partition)
    cluster_sizes = np.array([np.sum(partition == cluster) for cluster in unique_clusters], dtype=np.int64)
    ordered_clusters = unique_clusters[np.argsort(cluster_sizes, kind="stable")]

    new_partition = np.zeros_like(partition)
    for cluster_index, cluster_label in enumerate(ordered_clusters, start=1):
        new_partition[partition == cluster_label] = cluster_index
    return new_partition


def ind_mix(
    c: BapsData,
    npops: int,
    disp_text: bool = False,
    alpha: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.RandomState(1)

    state = _IndMixState(c, alpha, rng)
    partition_summary = np.full((30, 2), NEG_INF, dtype=np.float64)
    partition_summary[:, 0] = 0.0
    worst_logml = NEG_INF
    worst_index = 0
    logml_best = NEG_INF
    npops_best = npops
    partition_best = np.ones(c.rows.shape[0], dtype=np.int64)
    counts_best = None
    sumcounts_best = None
    pop_logml_best = None
    logdiff_best = None

    npops_table = np.atleast_1d(np.asarray(npops, dtype=np.int64))
    init_data = np.column_stack([c.data, np.arange(1, c.data.shape[0] + 1, dtype=np.int64)])

    for run_index, npops_now in enumerate(npops_table, start=1):
        if disp_text:
            print("-" * 51)
            print(f"Run {run_index}/{len(npops_table)}, maximum number of populations {int(npops_now)}.")

        initial_partition = admixture_initialization(init_data, int(npops_now), c.Z)
        sumcounts, counts, logml = initial_counts(initial_partition, c.data, int(npops_now), c.noalle, c.adjprior, alpha)
        state.partition = initial_partition.copy()
        state.counts = counts
        state.sumcounts = sumcounts
        state.pop_logml = state.compute_population_logml(np.arange(1, int(npops_now) + 1))
        state.logdiff = np.full((c.rows.shape[0], int(npops_now)), -np.inf, dtype=np.float64)

        n_round_types = 7
        tried = np.zeros(n_round_types, dtype=np.int64)
        round_types = [1, 1]
        stage = 1
        ready = False

        while not ready:
            changes = False

            for round_type in round_types:
                if tried[round_type - 1] == 1:
                    continue
                if round_type == 1:
                    moved = False
                    inds = rng.permutation(c.rows.shape[0]) + 1
                    for ind in inds:
                        current_pop = state.partition[ind - 1]
                        changes_now, diff_in_counts = state.calculate_move_deltas(ind)
                        best_index = int(np.argmax(changes_now)) + 1
                        max_change = float(changes_now[best_index - 1])
                        if current_pop != best_index and max_change > LOGML_EPS:
                            changes = True
                            moved = True
                            tried[:] = 0
                            state.update_global_variables(ind, best_index, diff_in_counts)
                            logml += max_change
                            if logml > worst_logml:
                                partition_summary, added = add_to_summary(logml, partition_summary, worst_index, state.partition)
                                if added:
                                    worst_index = int(np.argmin(partition_summary[:, 1]))
                                    worst_logml = float(partition_summary[worst_index, 1])
                    if not moved:
                        tried[round_type - 1] = 1

                elif round_type == 2:
                    max_change = 0.0
                    best_i1 = 1
                    best_i2 = 1
                    best_diff = None
                    for pop in range(1, state.counts.shape[2] + 1):
                        changes_now, diff_in_counts = state.calculate_merge_deltas(pop)
                        candidate_index = int(np.argmax(changes_now)) + 1
                        candidate = float(changes_now[candidate_index - 1])
                        if candidate > max_change:
                            max_change = candidate
                            best_i1 = pop
                            best_i2 = candidate_index
                            best_diff = diff_in_counts
                    if max_change > LOGML_EPS and best_diff is not None:
                        changes = True
                        tried[:] = 0
                        state.update_global_variables2(best_i1, best_i2, best_diff)
                        logml += max_change
                        if logml > worst_logml:
                            partition_summary, added = add_to_summary(logml, partition_summary, worst_index, state.partition)
                            if added:
                                worst_index = int(np.argmin(partition_summary[:, 1]))
                                worst_logml = float(partition_summary[worst_index, 1])
                    else:
                        tried[round_type - 1] = 1

                elif round_type in (3, 4):
                    max_change = 0.0
                    moving_inds = np.array([], dtype=np.int64)
                    target_pop = 1
                    for pop in range(1, state.counts.shape[2] + 1):
                        inds2 = np.where(state.partition == pop)[0] + 1
                        if inds2.size <= 2:
                            continue
                        dist2 = subset_distances(inds2, c.dist, c.rows.shape[0])
                        z2 = linkage_matlab(dist2)
                        npops2 = max(min(20, int(np.floor(inds2.size / 5))), 2) if round_type == 3 else 2
                        t2 = cluster_own(z2, npops2)
                        changes_now = state.calculate_split_deltas(t2, inds2, pop)
                        flat_index = int(np.argmax(changes_now))
                        candidate = float(changes_now.reshape(-1)[flat_index])
                        if candidate > max_change:
                            max_change = candidate
                            moving_pop2 = flat_index % npops2 + 1
                            moving_inds = inds2[t2 == moving_pop2]
                            target_pop = flat_index // npops2 + 1
                    if max_change > LOGML_EPS and moving_inds.size > 0:
                        changes = True
                        tried[:] = 0
                        rows = np.concatenate([np.arange(c.rows[ind - 1, 0], c.rows[ind - 1, 1] + 1) for ind in moving_inds])
                        diff_in_counts = compute_diff_in_counts(rows, state.counts.shape[0], state.counts.shape[1], c.data)
                        state.update_global_variables3(moving_inds, diff_in_counts, target_pop)
                        logml += max_change
                        if logml > worst_logml:
                            partition_summary, added = add_to_summary(logml, partition_summary, worst_index, state.partition)
                            if added:
                                worst_index = int(np.argmin(partition_summary[:, 1]))
                                worst_logml = float(partition_summary[worst_index, 1])
                    else:
                        tried[round_type - 1] = 1

                elif round_type in (5, 6):
                    j = 0
                    changed_now = False
                    pop_logml = state.pop_logml.copy()
                    partition_snapshot = state.partition.copy()
                    counts_snapshot = state.counts.copy()
                    sumcounts_snapshot = state.sumcounts.copy()
                    logdiff_snapshot = state.logdiff.copy()
                    pops = rng.permutation(state.counts.shape[2]) + 1
                    while j < state.counts.shape[2] and not changed_now:
                        j += 1
                        pop = int(pops[j - 1])
                        total_change = 0.0
                        inds = np.where(state.partition == pop)[0] + 1
                        if round_type == 5:
                            inds = inds[rng.permutation(len(inds))]
                        else:
                            inds = state.return_in_order(inds, pop)
                        i = 0
                        while len(inds) > 0 and i < len(inds):
                            ind = int(inds[i])
                            changes_now, diff_in_counts = state.calculate_move_deltas(ind)
                            changes_now[pop - 1] = NEG_INF
                            best_index = int(np.argmax(changes_now)) + 1
                            max_change = float(changes_now[best_index - 1])
                            state.update_global_variables(ind, best_index, diff_in_counts)
                            total_change += max_change
                            logml += max_change
                            if round_type == 6 and total_change > LOGML_EPS:
                                i = len(inds)
                            i += 1
                        if total_change > LOGML_EPS:
                            tried[:] = 0
                            changed_now = True
                            changes = True
                            if logml > worst_logml:
                                partition_summary, added = add_to_summary(logml, partition_summary, worst_index, state.partition)
                                if added:
                                    worst_index = int(np.argmin(partition_summary[:, 1]))
                                    worst_logml = float(partition_summary[worst_index, 1])
                        else:
                            state.partition = partition_snapshot
                            state.sumcounts = sumcounts_snapshot
                            state.pop_logml = pop_logml
                            state.counts = counts_snapshot
                            state.logdiff = logdiff_snapshot
                            logml -= total_change
                            tried[round_type - 1] = 1

                elif round_type == 7:
                    empty_pop, _ = state.find_empty_pop()
                    j = 0
                    pops = rng.permutation(state.counts.shape[2]) + 1
                    changed_now = False
                    if empty_pop == -1:
                        j = state.counts.shape[2]
                    while j < state.counts.shape[2]:
                        j += 1
                        pop = int(pops[j - 1])
                        inds2 = np.where(state.partition == pop)[0] + 1
                        if inds2.size <= 5:
                            continue
                        partition_snapshot = state.partition.copy()
                        sumcounts_snapshot = state.sumcounts.copy()
                        counts_snapshot = state.counts.copy()
                        pop_logml_snapshot = state.pop_logml.copy()
                        logdiff_snapshot = state.logdiff.copy()
                        dist2 = subset_distances(inds2, c.dist, c.rows.shape[0])
                        z2 = linkage_matlab(dist2)
                        t2 = cluster_own(z2, 2)
                        moving_inds = inds2[t2 == 1]
                        changes_now = state.calculate_split_deltas(t2, inds2, pop)
                        total_change = float(changes_now[0, empty_pop - 1])
                        rows = np.concatenate([np.arange(c.rows[ind - 1, 0], c.rows[ind - 1, 1] + 1) for ind in moving_inds])
                        diff_in_counts = compute_diff_in_counts(rows, state.counts.shape[0], state.counts.shape[1], c.data)
                        state.update_global_variables3(moving_inds, diff_in_counts, empty_pop)

                        changed = True
                        while changed:
                            changed = False
                            move_scores = state.calculate_pairwise_switch_deltas(inds2, pop, empty_pop)
                            best_index = int(np.argmax(move_scores))
                            max_change = float(move_scores[best_index])
                            moving_ind = int(inds2[best_index])
                            if state.partition[moving_ind - 1] == pop:
                                target = empty_pop
                            else:
                                target = pop
                            if max_change > LOGML_EPS:
                                rows = np.arange(c.rows[moving_ind - 1, 0], c.rows[moving_ind - 1, 1] + 1)
                                diff_in_counts = compute_diff_in_counts(rows, state.counts.shape[0], state.counts.shape[1], c.data)
                                state.update_global_variables3(np.array([moving_ind], dtype=np.int64), diff_in_counts, target)
                                changed = True
                                total_change += max_change

                        if total_change > LOGML_EPS:
                            changes = True
                            changed_now = True
                            logml += total_change
                            tried[:] = 0
                            if logml > worst_logml:
                                partition_summary, added = add_to_summary(logml, partition_summary, worst_index, state.partition)
                                if added:
                                    worst_index = int(np.argmin(partition_summary[:, 1]))
                                    worst_logml = float(partition_summary[worst_index, 1])
                            break

                        state.partition = partition_snapshot
                        state.sumcounts = sumcounts_snapshot
                        state.counts = counts_snapshot
                        state.pop_logml = pop_logml_snapshot
                        state.logdiff = logdiff_snapshot

                    if not changed_now:
                        tried[round_type - 1] = 1

            if not changes:
                stage += 1
                if stage > 5:
                    ready = True
            if not ready:
                if stage == 1:
                    round_types = [1]
                elif stage == 2:
                    round_types = [2, 1]
                elif stage == 3:
                    round_types = [5, 5, 7]
                elif stage == 4:
                    round_types = [4, 3, 1]
                else:
                    round_types = [6, 7, 2, 3, 4, 1]

        npops_now = state.remove_empty_populations()
        state.pop_logml = state.compute_population_logml(np.arange(1, npops_now + 1))

        if logml > logml_best:
            logml_best = logml
            npops_best = npops_now
            partition_best = state.partition.copy()
            counts_best = state.counts.copy()
            sumcounts_best = state.sumcounts.copy()
            pop_logml_best = state.pop_logml.copy()
            logdiff_best = state.logdiff.copy()

    state.partition = partition_best
    state.counts = counts_best
    state.sumcounts = sumcounts_best
    state.pop_logml = pop_logml_best
    state.logdiff = logdiff_best
    return float(logml_best), int(npops_best), partition_summary, partition_best


def new_get_distances(data: np.ndarray, rows_from_ind: int) -> tuple[np.ndarray, np.ndarray]:
    ninds = data.shape[0]
    pairs = [(a, b) for a in range(ninds - 1) for b in range(a + 1, ninds)]
    dist = np.zeros(len(pairs), dtype=np.float64)
    for index, (a, b) in enumerate(pairs):
        x = data[a]
        y = data[b]
        observed = (x > 0) & (y > 0)
        if not np.any(observed):
            dist[index] = 1.0
        else:
            dist[index] = float(np.mean(x[observed] != y[observed]))
    return linkage_matlab(dist), dist


def linkage_matlab(dist: np.ndarray) -> np.ndarray:
    if dist.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = scipy_linkage(dist, method="complete")
    z_mat = z[:, :3].copy()
    z_mat[:, 0:2] += 1.0
    return z_mat


def cluster_own(z: np.ndarray, nclust: int) -> np.ndarray:
    m = z.shape[0] + 1
    t = np.zeros(m, dtype=np.int64)
    if m <= nclust:
        return np.arange(1, m + 1, dtype=np.int64)
    if nclust == 1:
        return np.ones(m, dtype=np.int64)

    clsnum = 1
    for k in range(m - nclust, m - 1):
        i = int(z[k, 0])
        if i <= m:
            t[i - 1] = clsnum
            clsnum += 1
        elif i < (2 * m - nclust + 1):
            t = clusternum(z, t, i - m, clsnum)
            clsnum += 1

        j = int(z[k, 1])
        if j <= m:
            t[j - 1] = clsnum
            clsnum += 1
        elif j < (2 * m - nclust + 1):
            t = clusternum(z, t, j - m, clsnum)
            clsnum += 1
    return t


def clusternum(x: np.ndarray, t: np.ndarray, k: int, cluster_id: int) -> np.ndarray:
    m = x.shape[0] + 1
    pending = np.array([k], dtype=np.int64)
    while pending.size > 0:
        children = x[pending - 1, 0:2].reshape(-1).astype(np.int64)
        leaf_mask = children <= m
        t[children[leaf_mask] - 1] = cluster_id
        pending = children[~leaf_mask] - m
    return t


def initial_counts(
    partition: np.ndarray,
    data: np.ndarray,
    npops: int,
    noalle: np.ndarray,
    adjprior: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    nloci = data.shape[1]
    counts = np.zeros((int(np.max(noalle)), nloci, npops), dtype=np.float64)
    sumcounts = np.zeros((npops, nloci), dtype=np.float64)
    for pop_index in range(1, npops + 1):
        mask = partition == pop_index
        for locus_index in range(nloci):
            observations = np.where(mask & (data[:, locus_index] > 0))[0]
            sumcounts[pop_index - 1, locus_index] = len(observations)
            for allele_code in range(1, int(noalle[locus_index]) + 1):
                counts[allele_code - 1, locus_index, pop_index - 1] = np.sum(
                    data[observations, locus_index] == allele_code
                )
    logml = laske_loggis(counts, sumcounts, adjprior, alpha)
    return sumcounts, counts, float(logml)


def laske_loggis(counts: np.ndarray, sumcounts: np.ndarray, adjprior: np.ndarray, alpha: float) -> float:
    npops = counts.shape[2]
    logml = (
        np.sum(gammaln(alpha) * np.ones_like(sumcounts))
        + np.sum(gammaln(counts + np.repeat(adjprior[:, :, np.newaxis], npops, axis=2)))
        - npops * np.sum(gammaln(adjprior))
        - np.sum(gammaln(alpha + sumcounts))
    )
    return float(logml)


def admixture_initialization(data_matrix: np.ndarray, nclusters: int, z: np.ndarray) -> np.ndarray:
    n = int(np.max(data_matrix[:, -1]))
    t = cluster_own(z, nclusters)
    initial_partition = np.zeros(data_matrix.shape[0], dtype=np.int64)
    for ind in range(1, n + 1):
        here = np.where(data_matrix[:, -1] == ind)[0]
        initial_partition[here] = t[ind - 1]
    return initial_partition


def add_to_summary(
    logml: float,
    partition_summary: np.ndarray,
    worst_index: int,
    partition: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if np.any(np.abs(partition_summary[:, 1] - logml) < LOGML_EPS):
        return partition_summary, False
    partition_summary[worst_index, 0] = len(np.unique(partition))
    partition_summary[worst_index, 1] = logml
    return partition_summary, True


def subset_distances(inds2: np.ndarray, dist: np.ndarray, ninds: int) -> np.ndarray:
    ninds2 = len(inds2)
    indices = np.zeros(int(ninds2 * (ninds2 - 1) / 2), dtype=np.int64)
    row = 0
    for i in range(ninds2 - 1):
        for j in range(i + 1, ninds2):
            a = int(inds2[i])
            b = int(inds2[j])
            indices[row] = int((a - 1) * ninds - a / 2 * (a - 1) + (b - a)) - 1
            row += 1
    return dist[indices]


def _run_default_ind_mix(
    c: BapsData,
    alpha: float,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, int, np.ndarray]:
    for upper_bound in (5, 10, 20, 40):
        logml, npops, partition_summary, partition = ind_mix(c, upper_bound, False, alpha, rng)
        if npops != upper_bound or upper_bound == 40:
            return partition, npops, partition_summary
    raise RuntimeError("unreachable")


def _concatenate_baps_blocks(c_old: BapsData, c: BapsData) -> BapsData:
    if c.adjprior.shape[0] == c_old.adjprior.shape[0]:
        adjprior = np.concatenate([c_old.adjprior, c.adjprior], axis=1)
    elif c.adjprior.shape[0] > c_old.adjprior.shape[0]:
        diff = c.adjprior.shape[0] - c_old.adjprior.shape[0]
        aux = np.concatenate([c_old.adjprior, np.ones((diff, c_old.adjprior.shape[1]), dtype=np.float64)], axis=0)
        adjprior = np.concatenate([aux, c.adjprior], axis=1)
    else:
        diff = c_old.adjprior.shape[0] - c.adjprior.shape[0]
        aux = np.concatenate([c.adjprior, np.ones((diff, c.adjprior.shape[1]), dtype=np.float64)], axis=0)
        adjprior = np.concatenate([c_old.adjprior, aux], axis=1)
    return replace(
        c,
        adjprior=adjprior,
        priorTerm=c_old.priorTerm + c.priorTerm,
        data=np.concatenate([c_old.data, c.data], axis=1),
        noalle=np.concatenate([c_old.noalle, c.noalle]),
        snpPositions=np.concatenate([c_old.snpPositions, c.snpPositions]),
    )


class _IndMixState:
    def __init__(self, c: BapsData, alpha: float, rng: np.random.RandomState) -> None:
        self.data = c.data
        self.rows = c.rows
        self.adjprior = c.adjprior
        self.prior_term = c.priorTerm
        self.alpha = alpha
        self.rng = rng
        self.partition = np.zeros(c.rows.shape[0], dtype=np.int64)
        self.counts = np.zeros((int(np.max(c.noalle)), c.data.shape[1], 0), dtype=np.float64)
        self.sumcounts = np.zeros((0, c.data.shape[1]), dtype=np.float64)
        self.pop_logml = np.zeros(0, dtype=np.float64)
        self.logdiff = np.zeros((c.rows.shape[0], 0), dtype=np.float64)

    def compute_population_logml(self, pops: np.ndarray | list[int] | int) -> np.ndarray:
        pops_array = np.atleast_1d(np.asarray(pops, dtype=np.int64))
        y = self.counts.shape[1]
        result = (
            gammaln(self.alpha) * y
            + np.sum(gammaln(self.adjprior[:, :, np.newaxis] + self.counts[:, :, pops_array - 1]), axis=(0, 1))
            - np.sum(gammaln(self.alpha + self.sumcounts[pops_array - 1, :]), axis=1)
            - self.prior_term
        )
        return np.asarray(result, dtype=np.float64)

    def calculate_move_deltas(self, ind: int) -> tuple[np.ndarray, np.ndarray]:
        npops = self.counts.shape[2]
        changes = self.logdiff[ind - 1, :].copy()
        i1 = int(self.partition[ind - 1])
        i1_logml = float(self.pop_logml[i1 - 1])
        changes[i1 - 1] = 0.0

        rows = np.arange(self.rows[ind - 1, 0], self.rows[ind - 1, 1] + 1)
        diff_in_counts = compute_diff_in_counts(rows, self.counts.shape[0], self.counts.shape[1], self.data)
        diff_in_sumcounts = np.sum(diff_in_counts, axis=0)

        self.counts[:, :, i1 - 1] -= diff_in_counts
        self.sumcounts[i1 - 1, :] -= diff_in_sumcounts
        new_i1_logml = float(self.compute_population_logml(i1)[0])
        self.counts[:, :, i1 - 1] += diff_in_counts
        self.sumcounts[i1 - 1, :] += diff_in_sumcounts

        i2 = np.where(np.isneginf(changes))[0] + 1
        i2 = i2[i2 != i1]
        if i2.size > 0:
            i2_logml = self.pop_logml[i2 - 1]
            self.counts[:, :, i2 - 1] += diff_in_counts[:, :, np.newaxis]
            self.sumcounts[i2 - 1, :] += diff_in_sumcounts
            new_i2_logml = self.compute_population_logml(i2)
            self.counts[:, :, i2 - 1] -= diff_in_counts[:, :, np.newaxis]
            self.sumcounts[i2 - 1, :] -= diff_in_sumcounts
            changes[i2 - 1] = new_i1_logml - i1_logml + new_i2_logml - i2_logml

        self.logdiff[ind - 1, :] = changes
        return changes, diff_in_counts

    def update_global_variables(self, ind: int, i2: int, diff_in_counts: np.ndarray) -> None:
        i1 = int(self.partition[ind - 1])
        self.partition[ind - 1] = i2

        self.counts[:, :, i1 - 1] -= diff_in_counts
        self.counts[:, :, i2 - 1] += diff_in_counts
        diff_sum = np.sum(diff_in_counts, axis=0)
        self.sumcounts[i1 - 1, :] -= diff_sum
        self.sumcounts[i2 - 1, :] += diff_sum
        self.pop_logml[np.array([i1, i2]) - 1] = self.compute_population_logml(np.array([i1, i2]))
        self.logdiff[:, np.array([i1, i2]) - 1] = -np.inf
        idx = np.concatenate([np.where(self.partition == i1)[0], np.where(self.partition == i2)[0]])
        self.logdiff[idx, :] = -np.inf

    def calculate_merge_deltas(self, i1: int) -> tuple[np.ndarray, np.ndarray]:
        npops = self.counts.shape[2]
        changes = np.zeros(npops, dtype=np.float64)
        i1_logml = float(self.pop_logml[i1 - 1])
        inds = np.where(self.partition == i1)[0] + 1
        if inds.size == 0:
            return changes, np.zeros(self.counts.shape[:2], dtype=np.float64)
        rows = np.concatenate([np.arange(self.rows[ind - 1, 0], self.rows[ind - 1, 1] + 1) for ind in inds])
        diff_in_counts = compute_diff_in_counts(rows, self.counts.shape[0], self.counts.shape[1], self.data)
        diff_in_sumcounts = np.sum(diff_in_counts, axis=0)

        self.counts[:, :, i1 - 1] -= diff_in_counts
        self.sumcounts[i1 - 1, :] -= diff_in_sumcounts
        new_i1_logml = float(self.compute_population_logml(i1)[0])
        self.counts[:, :, i1 - 1] += diff_in_counts
        self.sumcounts[i1 - 1, :] += diff_in_sumcounts

        i2 = np.array([pop for pop in range(1, npops + 1) if pop != i1], dtype=np.int64)
        i2_logml = self.pop_logml[i2 - 1]
        self.counts[:, :, i2 - 1] += diff_in_counts[:, :, np.newaxis]
        self.sumcounts[i2 - 1, :] += diff_in_sumcounts
        new_i2_logml = self.compute_population_logml(i2)
        self.counts[:, :, i2 - 1] -= diff_in_counts[:, :, np.newaxis]
        self.sumcounts[i2 - 1, :] -= diff_in_sumcounts
        changes[i2 - 1] = new_i1_logml - i1_logml + new_i2_logml - i2_logml
        return changes, diff_in_counts

    def update_global_variables2(self, i1: int, i2: int, diff_in_counts: np.ndarray) -> None:
        inds = np.where(self.partition == i1)[0]
        self.partition[inds] = i2
        self.counts[:, :, i1 - 1] -= diff_in_counts
        self.counts[:, :, i2 - 1] += diff_in_counts
        diff_sum = np.sum(diff_in_counts, axis=0)
        self.sumcounts[i1 - 1, :] -= diff_sum
        self.sumcounts[i2 - 1, :] += diff_sum
        self.pop_logml[i1 - 1] = 0.0
        self.pop_logml[i2 - 1] = self.compute_population_logml(i2)[0]
        self.logdiff[:, np.array([i1, i2]) - 1] = -np.inf
        idx = np.concatenate([np.where(self.partition == i1)[0], np.where(self.partition == i2)[0]])
        self.logdiff[idx, :] = -np.inf

    def calculate_split_deltas(self, t2: np.ndarray, inds2: np.ndarray, i1: int) -> np.ndarray:
        npops = self.counts.shape[2]
        npops2 = len(np.unique(t2))
        changes = np.zeros((npops2, npops), dtype=np.float64)
        i1_logml = float(self.pop_logml[i1 - 1])
        for pop2 in range(1, npops2 + 1):
            inds = inds2[t2 == pop2]
            if inds.size == 0:
                continue
            rows = np.concatenate([np.arange(self.rows[ind - 1, 0], self.rows[ind - 1, 1] + 1) for ind in inds])
            diff_in_counts = compute_diff_in_counts(rows, self.counts.shape[0], self.counts.shape[1], self.data)
            diff_in_sumcounts = np.sum(diff_in_counts, axis=0)
            self.counts[:, :, i1 - 1] -= diff_in_counts
            self.sumcounts[i1 - 1, :] -= diff_in_sumcounts
            new_i1_logml = float(self.compute_population_logml(i1)[0])
            self.counts[:, :, i1 - 1] += diff_in_counts
            self.sumcounts[i1 - 1, :] += diff_in_sumcounts

            i2 = np.array([pop for pop in range(1, npops + 1) if pop != i1], dtype=np.int64)
            i2_logml = self.pop_logml[i2 - 1]
            self.counts[:, :, i2 - 1] += diff_in_counts[:, :, np.newaxis]
            self.sumcounts[i2 - 1, :] += diff_in_sumcounts
            new_i2_logml = self.compute_population_logml(i2)
            self.counts[:, :, i2 - 1] -= diff_in_counts[:, :, np.newaxis]
            self.sumcounts[i2 - 1, :] -= diff_in_sumcounts
            changes[pop2 - 1, i2 - 1] = new_i1_logml - i1_logml + new_i2_logml - i2_logml
        return changes

    def calculate_pairwise_switch_deltas(self, inds: np.ndarray, i1: int, i2: int) -> np.ndarray:
        ninds = len(inds)
        changes = np.zeros(ninds, dtype=np.float64)
        i1_logml = float(self.pop_logml[i1 - 1])
        i2_logml = float(self.pop_logml[i2 - 1])
        for idx, ind in enumerate(inds):
            if self.partition[ind - 1] == i1:
                pop1 = i1
                pop2 = i2
            else:
                pop1 = i2
                pop2 = i1
            rows = np.arange(self.rows[ind - 1, 0], self.rows[ind - 1, 1] + 1)
            diff_in_counts = compute_diff_in_counts(rows, self.counts.shape[0], self.counts.shape[1], self.data)
            diff_in_sumcounts = np.sum(diff_in_counts, axis=0)
            self.counts[:, :, pop1 - 1] -= diff_in_counts
            self.sumcounts[pop1 - 1, :] -= diff_in_sumcounts
            self.counts[:, :, pop2 - 1] += diff_in_counts
            self.sumcounts[pop2 - 1, :] += diff_in_sumcounts
            changes[idx] = np.sum(self.compute_population_logml(np.array([i1, i2])))
            self.counts[:, :, pop1 - 1] += diff_in_counts
            self.sumcounts[pop1 - 1, :] += diff_in_sumcounts
            self.counts[:, :, pop2 - 1] -= diff_in_counts
            self.sumcounts[pop2 - 1, :] -= diff_in_sumcounts
        return changes - i1_logml - i2_logml

    def update_global_variables3(self, moving: np.ndarray, diff_in_counts: np.ndarray, i2: int) -> None:
        i1 = int(self.partition[int(moving[0]) - 1])
        self.partition[np.asarray(moving, dtype=np.int64) - 1] = i2
        self.counts[:, :, i1 - 1] -= diff_in_counts
        self.counts[:, :, i2 - 1] += diff_in_counts
        diff_sum = np.sum(diff_in_counts, axis=0)
        self.sumcounts[i1 - 1, :] -= diff_sum
        self.sumcounts[i2 - 1, :] += diff_sum
        self.pop_logml[np.array([i1, i2]) - 1] = self.compute_population_logml(np.array([i1, i2]))
        self.logdiff[:, np.array([i1, i2]) - 1] = -np.inf
        idx = np.concatenate([np.where(self.partition == i1)[0], np.where(self.partition == i2)[0]])
        self.logdiff[idx, :] = -np.inf

    def remove_empty_populations(self) -> int:
        not_empty = np.where(np.any(self.sumcounts != 0, axis=1))[0] + 1
        self.counts = self.counts[:, :, not_empty - 1]
        self.sumcounts = self.sumcounts[not_empty - 1, :]
        self.logdiff = self.logdiff[:, not_empty - 1]
        for new_index, old_index in enumerate(not_empty, start=1):
            self.partition[self.partition == old_index] = new_index
        return len(not_empty)

    def return_in_order(self, inds: np.ndarray, pop: int) -> np.ndarray:
        ranking = np.zeros((len(inds), 2), dtype=np.float64)
        ranking[:, 0] = inds
        for index, ind in enumerate(inds):
            rows = np.arange(self.rows[ind - 1, 0], self.rows[ind - 1, 1] + 1)
            diff_in_counts = compute_diff_in_counts(rows, self.counts.shape[0], self.counts.shape[1], self.data)
            diff_in_sumcounts = np.sum(diff_in_counts, axis=0)
            self.counts[:, :, pop - 1] -= diff_in_counts
            self.sumcounts[pop - 1, :] -= diff_in_sumcounts
            ranking[index, 1] = self.compute_population_logml(pop)[0]
            self.counts[:, :, pop - 1] += diff_in_counts
            self.sumcounts[pop - 1, :] += diff_in_sumcounts
        return ranking[np.argsort(ranking[:, 1]), 0].astype(np.int64)[::-1]

    def find_empty_pop(self) -> tuple[int, np.ndarray]:
        pops = np.unique(self.partition)
        if len(pops) == self.counts.shape[2]:
            return -1, pops
        pop_diff = np.diff(np.concatenate([[0], pops, [self.counts.shape[2] + 1]]))
        return int(np.min(np.where(pop_diff > 1)[0])), pops
