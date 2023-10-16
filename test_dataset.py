import torch
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset
from bigearthnet import BigEarthNetHDF5
from torchgeo.datasets import BigEarthNet
from pathlib import Path
from typing import Literal, Optional, Callable, List, Tuple, Dict

def compute_full_read_time(dataset: Dataset) -> float:
    start = time.time()
    print(f'Starting read at {start}')
    count = 0
    for d in tqdm(dataset):
        count += 1
    end = time.time()
    print(f'End at {end} - read {count} elements')
    print(f'Time taken: {end - start} seconds')
    return

def generate_idx(start: int, end: int, n: int, strategy: Literal['uniform', 'random'], seed: Optional[int]=None) -> np.ndarray:
    if not seed is None:
        np.random.seed(seed)
    if strategy == 'random':
        res = np.random.randint(start, end + 1, size=n)
    else:
        res = np.linspace(start, end, endpoint=True, dtype=int)
    return res

def to_tensor(t) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(t)
    return t

def verify(d1: Dataset, d2: Dataset, indexes: np.ndarray) -> bool:
    for idx in indexes:
        v1 = d1[idx]
        v2 = d2[idx]

        img1, lbl1 = to_tensor(v1['image']), to_tensor(v1['label'])
        img2, lbl2 = to_tensor(v2['image']), to_tensor(v2['label'])

        img_eq = (img1 == img2).all()
        lbl_eq = (lbl1 == lbl2).all()

        if (img_eq and lbl_eq) != True:
            return False
    return True

def verify_datasets(
    d1: Dataset,
    d2: Dataset,
    block_size: int,
    num_samples: int=100,
    distrib: Literal['uniform', 'random']='uniform',
    seed: Optional[int]=None
) -> bool:
    if len(d1) != len(d2):
        return False
    total = len(d1)
    total_blocks = math.ceil(total / block_size)
    for block_number in range(total_blocks):
        start_idx = block_number * block_size
        end_idx = min((block_number + 1) * block_size - 1, total - 1)
        indexes = generate_idx(start_idx, end_idx, num_samples, distrib, seed)

        verification = verify(d1, d2, indexes)
        if not verification:
            return False
        print(f'Passed test on {num_samples} on block {block_number}/{total_blocks}')
    return True

if __name__ == '__main__':
    dataset_args = {
        'root': 'PATH',
        'download': False,
        'checksum': False,
        'bands': 's2',
        'num_classes': 43
    }
    dataset_2_args = dataset_args.copy()
    dataset_2_args['root'] = 'PATH'
    dataset_2_args['train_filename'] = 'bigearthnet_hdf5_train.csv'
    dataset_2_args['val_filename'] = 'bigearthnet_hdf5_val.csv'
    dataset_2_args['test_filename'] = 'bigearthnet_hdf5_test.csv'

    dataset_original = BigEarthNet(**dataset_args)
    dataset_hdf5 = BigEarthNetHDF5(**dataset_2_args)

    # print(f'HDF5')
    # compute_full_read_time(dataset_hdf5)
    # print(f'Torchgeo')
    # compute_full_read_time(dataset_original)
    verification = verify_datasets(dataset_original, dataset_hdf5, block_size=15_000, num_samples=1000, distrib='random')
    print(verification)
