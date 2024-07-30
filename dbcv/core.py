import multiprocessing
import typing as t
import itertools
import functools

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import scipy.spatial.distance
import scipy.sparse.csgraph
import scipy.stats
import mpmath
from tqdm import tqdm #Jonas added progress bar
from . import reference_prim_mst


_MP = mpmath.mp.clone()

#### New Functions added by Jonas K below ####
def compute_distances_on_the_fly(X: npt.NDArray[np.float64], metric: str, batch_size: int = 1000):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = X[i:end]
        yield i, scipy.spatial.distance.cdist(batch, X, metric=metric)

def get_core_distances(dists: npt.NDArray[np.float64], d: int, enable_dynamic_precision: bool) -> npt.NDArray[np.float64]:
    n = dists.shape[0]
    orig_dists_dtype = dists.dtype

    if enable_dynamic_precision:
        dists = np.asarray(_MP.matrix(dists), dtype=object).reshape(*dists.shape)

    core_dists = np.power(dists, -d).sum(axis=-1) / (n - 1)

    if not enable_dynamic_precision:
        np.clip(core_dists, a_min=0.0, a_max=1e12, out=core_dists)

    np.power(core_dists, -1.0 / d, out=core_dists)

    if enable_dynamic_precision:
        core_dists = np.asfarray(core_dists, dtype=orig_dists_dtype)

    return core_dists

def process_cluster(cls_inds: npt.NDArray[np.int32], X: npt.NDArray[np.float64], d: int, metric: str,
                    enable_dynamic_precision: bool, use_original_mst_implementation: bool, batch_size: int):
    cls_X = X[cls_inds]
    n_cls = cls_X.shape[0]

    mutual_reach_dists = np.full((n_cls, n_cls), np.inf)
    core_dists = np.zeros(n_cls)

    for i, batch_dists in compute_distances_on_the_fly(cls_X, metric, batch_size):
        np.maximum(batch_dists, 1e-12, out=batch_dists)
        np.fill_diagonal(batch_dists[max(0, i-batch_size):], np.inf)

        batch_core_dists = get_core_distances(batch_dists, d, enable_dynamic_precision)
        core_dists[i:i+batch_size] = batch_core_dists

        np.maximum(batch_dists, batch_core_dists[:, np.newaxis], out=batch_dists)
        np.maximum(batch_dists, core_dists, out=batch_dists)

        mutual_reach_dists[i:i+batch_size] = batch_dists
        mutual_reach_dists[:, i:i+batch_size] = batch_dists.T

    internal_node_inds, internal_edge_weights = get_internal_objects(
        mutual_reach_dists, use_original_mst_implementation=use_original_mst_implementation
    )
    dsc = float(internal_edge_weights.max())
    internal_core_dists = core_dists[internal_node_inds]
    internal_node_inds = cls_inds[internal_node_inds]
    return (dsc, internal_core_dists, internal_node_inds)

##### New functions added by Jonas above ###

def get_subarray(
    arr: npt.NDArray[np.float64],
    /,
    inds_a: t.Optional[npt.NDArray[np.int32]] = None,
    inds_b: t.Optional[npt.NDArray[np.int32]] = None,
) -> npt.NDArray[np.float64]:
    if inds_a is None:
        return arr
    if inds_b is None:
        inds_b = inds_a
    inds_a_mesh, inds_b_mesh = np.meshgrid(inds_a, inds_b)
    return arr[inds_a_mesh, inds_b_mesh].T


def get_internal_objects(mutual_reach_dists: npt.NDArray[np.float64], use_original_mst_implementation: bool) -> npt.NDArray[np.float64]:
    if use_original_mst_implementation:
        mutual_reach_dists = np.copy(mutual_reach_dists)
        np.fill_diagonal(mutual_reach_dists, 0.0)
        mst = reference_prim_mst.prim_mst(mutual_reach_dists)

    else:
        mst = scipy.sparse.csgraph.minimum_spanning_tree(mutual_reach_dists)
        mst = mst.toarray()
        mst += mst.T

    is_mst_edges = (mst > 0.0).astype(int, copy=False)

    internal_node_inds = is_mst_edges.sum(axis=0) > 1
    internal_node_inds = np.flatnonzero(internal_node_inds)

    internal_edge_weights = get_subarray(mst, inds_a=internal_node_inds)

    return internal_node_inds, internal_edge_weights

#Removed from the original implementation
# def compute_cluster_core_distance(dists: npt.NDArray[np.float64], d: int, enable_dynamic_precision: bool) -> npt.NDArray[np.float64]:
#     n, _ = dists.shape
#     orig_dists_dtype = dists.dtype

#     if enable_dynamic_precision:
#         dists = np.asarray(_MP.matrix(dists), dtype=object).reshape(*dists.shape)

#     core_dists = np.power(dists, -d).sum(axis=-1, keepdims=True) / (n - 1)

#     if not enable_dynamic_precision:
#         np.clip(core_dists, a_min=0.0, a_max=1e12, out=core_dists)

#     np.power(core_dists, -1.0 / d, out=core_dists)

#     if enable_dynamic_precision:
#         core_dists = np.asfarray(core_dists, dtype=orig_dists_dtype)

#     return core_dists


# def compute_mutual_reach_dists(
#     dists: npt.NDArray[np.float64],
#     d: float,
#     enable_dynamic_precision: bool,
# ) -> npt.NDArray[np.float64]:
#     core_dists = compute_cluster_core_distance(d=d, dists=dists, enable_dynamic_precision=enable_dynamic_precision)
#     mutual_reach_dists = dists.copy()
#     np.maximum(mutual_reach_dists, core_dists, out=mutual_reach_dists)
#     np.maximum(mutual_reach_dists, core_dists.T, out=mutual_reach_dists)
#     return (core_dists, mutual_reach_dists)


# def fn_density_sparseness(
#     cls_inds: npt.NDArray[np.int32],
#     dists: npt.NDArray[np.float64],
#     d: int,
#     enable_dynamic_precision: bool,
#     use_original_mst_implementation: bool,
# ) -> t.Tuple[float, npt.NDArray[np.float32], npt.NDArray[np.int32]]:
#     (core_dists, mutual_reach_dists) = compute_mutual_reach_dists(dists=dists, d=d, enable_dynamic_precision=enable_dynamic_precision)
#     internal_node_inds, internal_edge_weights = get_internal_objects(
#         mutual_reach_dists, use_original_mst_implementation=use_original_mst_implementation
#     )
#     dsc = float(internal_edge_weights.max())
#     internal_core_dists = core_dists[internal_node_inds]
#     internal_node_inds = cls_inds[internal_node_inds]
#     return (dsc, internal_core_dists, internal_node_inds)


#New fn_density_separation added by Jonas K
def fn_density_separation(
    cls_i: int,
    cls_j: int,
    X_i: npt.NDArray[np.float64],
    X_j: npt.NDArray[np.float64],
    internal_core_dists_i: npt.NDArray[np.float64],
    internal_core_dists_j: npt.NDArray[np.float64],
    metric: str,
) -> t.Tuple[int, int, float]:
    dists = scipy.spatial.distance.cdist(X_i, X_j, metric=metric)
    sep = np.maximum(dists, internal_core_dists_i[:, np.newaxis])
    sep = np.maximum(sep, internal_core_dists_j)
    dspc_ij = float(sep.min()) if sep.size else np.inf
    return (cls_i, cls_j, dspc_ij)

def _check_duplicated_samples(X: npt.NDArray[np.float64], threshold: float = 1e-9):
    if X.shape[0] <= 1:
        return

    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    nn.fit(X)
    dists, _ = nn.kneighbors(return_distance=True)

    if np.any(dists < threshold):
        raise ValueError("Duplicated samples have been found in X.")


def _convert_singleton_clusters_to_noise(y: npt.NDArray[np.int32], noise_id: int) -> npt.NDArray[np.int32]:
    """Cast clusters containing a single instance as noise."""
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)
    singleton_clusters = cluster_ids[cluster_sizes == 1]

    if singleton_clusters.size == 0:
        return y

    return np.where(np.isin(y, singleton_clusters), noise_id, y)

#Adapted dbcv by Jonas K
def dbcv(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int32],
    metric: str = "sqeuclidean",
    noise_id: int = -1,
    check_duplicates: bool = True,
    n_processes: t.Union[int, str] = "auto",
    enable_dynamic_precision: bool = False,
    bits_of_precision: int = 512,
    use_original_mst_implementation: bool = False,
    batch_size: int = 1000,
) -> float:
    X = np.asfarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = np.asarray(y, dtype=int)

    n, d = X.shape  # NOTE: 'n' must be calculated before removing noise.

    if n != y.size:
        raise ValueError(f"Mismatch in {X.shape[0]=} and {y.size=} dimensions.")

    y = _convert_singleton_clusters_to_noise(y, noise_id=noise_id)

    non_noise_inds = y != noise_id
    X = X[non_noise_inds, :]
    y = y[non_noise_inds]

    if y.size == 0:
        return 0.0

    y = scipy.stats.rankdata(y, method="dense") - 1
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)

    if check_duplicates:
        _check_duplicated_samples(X)

    dscs = np.zeros(cluster_ids.size, dtype=float)
    min_dspcs = np.full(cluster_ids.size, fill_value=np.inf)
    internal_objects_per_cls: t.Dict[int, npt.NDArray[np.int32]] = {}
    internal_core_dists_per_cls: t.Dict[int, npt.NDArray[np.float32]] = {}

    cls_inds = [np.flatnonzero(y == cls_id) for cls_id in cluster_ids]

    if n_processes == "auto":
        n_processes = 4 if y.size > 500 else 1

    with _MP.workprec(bits_of_precision), multiprocessing.Pool(processes=min(n_processes, cluster_ids.size)) as ppool:
        process_cluster_ = functools.partial(
            process_cluster,
            X=X,
            d=d,
            metric=metric,
            enable_dynamic_precision=enable_dynamic_precision,
            use_original_mst_implementation=use_original_mst_implementation,
            batch_size=batch_size,
        )

        for cls_id, (dsc, internal_core_dists, internal_node_inds) in enumerate(ppool.imap(process_cluster_, cls_inds)):
            internal_objects_per_cls[cls_id] = internal_node_inds
            internal_core_dists_per_cls[cls_id] = internal_core_dists
            dscs[cls_id] = dsc

    n_cls_pairs = (cluster_ids.size * (cluster_ids.size - 1)) // 2

    if n_cls_pairs > 0:
        with _MP.workprec(bits_of_precision), multiprocessing.Pool(processes=min(n_processes, n_cls_pairs)) as ppool:
            args = [
                (
                    cls_i,
                    cls_j,
                    X[internal_objects_per_cls[cls_i]],
                    X[internal_objects_per_cls[cls_j]],
                    internal_core_dists_per_cls[cls_i],
                    internal_core_dists_per_cls[cls_j],
                    metric,
                )
                for cls_i, cls_j in itertools.combinations(cluster_ids, 2)
            ]

            for cls_i, cls_j, dspc_ij in ppool.starmap(fn_density_separation, args):
                min_dspcs[cls_i] = min(min_dspcs[cls_i], dspc_ij)
                min_dspcs[cls_j] = min(min_dspcs[cls_j], dspc_ij)

    np.nan_to_num(min_dspcs, copy=False, posinf=1e12)
    vcs = (min_dspcs - dscs) / (1e-12 + np.maximum(min_dspcs, dscs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    dbcv = float(np.sum(vcs * cluster_sizes)) / n

    return dbcv
