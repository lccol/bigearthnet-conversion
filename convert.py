import torch
import numpy as np
import argparse
import pandas as pd
import math
import h5py

from collections import defaultdict

from tqdm import tqdm
from torchgeo.datasets import BigEarthNet
from pathlib import Path

from typing import Tuple

def parse():
    parser = argparse.ArgumentParser(description='Compress bigearthnet into HDF5 representation')
    parser.add_argument('-n', type=int, default=50_000, help='Max numnber of entries per file')
    parser.add_argument('--dataset-path', type=str, required=True, help='Master folder containing the dataset')
    parser.add_argument('--output-path', type=str, required=True, help='Output folder in which the results is saved')
    parser.add_argument('--dev', action='store_true', help='If true, writes just 10 entries per split')
    return parser.parse_args()

def verify_empty_folder(dst: Path) -> None:
    return

def compute_block_size(idx: int, max_block_size: int, dataset_size: int, dev: bool=False) -> Tuple[int, int, int]:
    # return curr_block, curr_block_size, offset
    curr_block = math.floor(idx / max_block_size)
    block_start_idx = curr_block * max_block_size
    remaining_records = dataset_size - block_start_idx + 1
    curr_block_size = min(remaining_records, max_block_size)
    offset = idx - block_start_idx
    if dev:
        curr_block_size = min(curr_block_size, 10)
    return curr_block, curr_block_size, offset

if __name__ == '__main__':
    args = parse()
    
    N = args.n
    DATASET_PATH = Path(args.dataset_path)
    OUTPUT_PATH = Path(args.output_path)
    DEV = args.dev

    bigearthnet_root = Path(DATASET_PATH)

    if not OUTPUT_PATH.is_dir():
        OUTPUT_PATH.mkdir(parents=True)

    splits = ['train', 'val', 'test']
    
    for spl in splits:
        prev_block = -1
        csv_path = DATASET_PATH / f'bigearthnet-{spl}.csv'
        assert csv_path.is_file(), f'File not found {csv_path}'
        df = pd.read_csv(csv_path, header=None)
        total_size = df.shape[0]

        print(f'BigEarthNet root folder: {str(bigearthnet_root)}')

        dataset19 = BigEarthNet(root=str(bigearthnet_root), split=spl, num_classes=19, download=False, checksum=False, bands='s2')
        dataset43 = BigEarthNet(root=str(bigearthnet_root), split=spl, num_classes=43, download=False, checksum=False, bands='s2')

        file_mapping = defaultdict(list)

        filep = None
        print(f'Start processing split {spl}')
        for idx, row in tqdm(df.iterrows(), total=total_size):
            curr_block, block_size, offset = compute_block_size(idx, N, total_size, DEV)
            if curr_block != prev_block:
                if not filep is None:
                    filep.close()
                hdf5_file = f'bigearthnet_{spl}_p{curr_block}.hdf5'
                filep = h5py.File(OUTPUT_PATH / hdf5_file, 'w')
                dataset_images = filep.create_dataset('images', shape=(block_size, 12, 120, 120), dtype=np.float32)
                dataset_labels19 = filep.create_dataset('labels19', shape=(block_size, 19), dtype=np.int64)
                dataset_labels43 = filep.create_dataset('labels43', shape=(block_size, 43), dtype=np.int64)
            d = dataset19[idx]
            img, label19 = d['image'], d['label']
            label43 = dataset43[idx]['label']

            dataset_images[offset, :, :, :] = img
            dataset_labels19[offset, :] = label19
            dataset_labels43[offset, :] = label43

            file_mapping['s2_folder'].append(row[0])
            file_mapping['s2_hdf5_file'].append(hdf5_file)
            file_mapping['index'].append(offset)
            prev_block = curr_block

            if DEV and idx >= 10:
                break

        file_mapping_df = pd.DataFrame(file_mapping)
        file_mapping_df.to_csv(OUTPUT_PATH / f'bigearthnet_hdf5_{spl}.csv', index=False)

    filep.close()
