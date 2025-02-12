"""
    This script provides the functionality to generate metadata.npz files necessary
    for load_balancing the DataLoader.
    
    1. Copy from: https://github.com/Open-Catalyst-Project/ocp/blob/09f0d3bdf4c9154c2f11105eb5d61e1cc0d8c638/scripts/make_lmdb_sizes.py
    2. Used for generating sizes of data for S2EF task. Since the data is generated with
    PyG 2+, the code for generating sizes of data should be modified.
    
"""


import argparse
import multiprocessing as mp
import os
import warnings

import numpy as np
from tqdm import tqdm

#from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset
from lmdb_dataset import SinglePointLmdbDatasetV2, TrajectoryLmdbDatasetV2 


def get_data(index):
    data = dataset[index]
    natoms = data.natoms
    neighbors = None
    if hasattr(data, "edge_index") and data.edge_index is not None:
        neighbors = data.edge_index.shape[1]

    return index, natoms, neighbors


def main(args):
    path = args.data_path
    global dataset
    if os.path.isdir(path):
        dataset = TrajectoryLmdbDatasetV2({"src": path})
        outpath = os.path.join(path, "metadata.npz")
    elif os.path.isfile(path):
        dataset = SinglePointLmdbDatasetV2({"src": path})
        outpath = os.path.join(os.path.dirname(path), "metadata.npz")

    indices = range(len(dataset))

    pool = mp.Pool(args.num_workers)
    outputs = list(tqdm(pool.imap(get_data, indices), total=len(indices)))

    indices = []
    natoms = []
    neighbors = []
    for i in outputs:
        indices.append(i[0])
        natoms.append(i[1])
        neighbors.append(i[2])

    _sort = np.argsort(indices)
    sorted_natoms = np.array(natoms, dtype=np.int32)[_sort]
    if None in neighbors:
        warnings.warn(
            f"edge_index information not found, {outpath} only supports atom-wise load balancing."
        )
        np.savez(outpath, natoms=sorted_natoms)
    else:
        sorted_neighbors = np.array(neighbors, dtype=np.int32)[_sort]
        np.savez(outpath, natoms=sorted_natoms, neighbors=sorted_neighbors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        help="Path to S2EF directory or IS2R* .lmdb file",
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Num of workers to parallelize across",
    )
    args = parser.parse_args()
    main(args)
