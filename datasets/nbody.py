import torch
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
from torch.utils.data import Subset
import os


class NBodyDataset(InMemoryDataset):
    def __init__(self, root, dataset_arg, transform=None, pre_transform=None):
        self.dataset_arg = dataset_arg
        super(NBodyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            f'loc_{self.dataset_arg}.npy',
            f'vel_{self.dataset_arg}.npy',
            f'edges_{self.dataset_arg}.npy',
            f'charges_{self.dataset_arg}.npy'
        ]

    @property
    def processed_file_names(self):
        return [f'data_{self.dataset_arg}.pt']

    def download(self):
        pass

    def process(self):
        loc = np.load(os.path.join(self.raw_dir, f'loc_{self.dataset_arg}.npy'))
        vel = np.load(os.path.join(self.raw_dir, f'vel_{self.dataset_arg}.npy'))
        edges = np.load(os.path.join(self.raw_dir, f'edges_{self.dataset_arg}.npy'))
        charges = np.load(os.path.join(self.raw_dir, f'charges_{self.dataset_arg}.npy'))

        data_list = []
        for i in range(loc.shape[0]):
            data = Data(
                pos=torch.from_numpy(loc[i]).float(),
                vel=torch.from_numpy(vel[i]).float(),
                edge_index=torch.from_numpy(edges[i]).long().t().contiguous(),
                charges=torch.from_numpy(charges[i]).float()
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    
    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total})."
    )
    if total < dset_len:
        print(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int)
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(
    dataset_len,
    train_size,
    val_size,
    test_size,
    seed,
    filename=None,  # path to save split index
    splits=None,
    order=None,
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


def get_nbody_datasets(root, dataset_arg, train_size, val_size, test_size, seed):
    all_dataset = NBodyDataset(root, dataset_arg)

    idx_train, idx_val, idx_test = make_splits(
        len(all_dataset),
        train_size, val_size, test_size,
        seed,
        filename=os.path.join(root, 'splits.npz'),
        splits=None
    )

    train_dataset = Subset(all_dataset, idx_train)
    val_dataset = Subset(all_dataset, idx_val)
    test_dataset = Subset(all_dataset, idx_test)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader

    _root_path = './nbody_data'
    _dataset_arg = 'train_charged5_initvel1tiny'
    train_dataset, val_dataset, test_dataset = get_nbody_datasets(
        root=_root_path,
        dataset_arg=_dataset_arg,
        train_size=800, val_size=100, test_size=100,
        seed=42
    )

    print('Training set size:   {}'.format(len(train_dataset)))
    print('Validation set size: {}'.format(len(val_dataset)))
    print('Testing set size:    {}'.format(len(test_dataset)))

    print(train_dataset[2])

    train_loader = DataLoader(train_dataset, batch_size=8)
    for i, data in enumerate(train_loader):
        print(data)
        print(data.charges)
        break
