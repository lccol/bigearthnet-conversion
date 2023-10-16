import torch
import numpy as np
import pandas as pd
import h5py

from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, Tuple, Dict, List, Optional, Callable, Literal, LiteralString

class BigEarthNetHDF5(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: Literal['train', 'val', 'test']='train',
        bands: Literal['s2', 's1', 'all']='s2',
        num_classes: Literal[19, 43]=19,
        transforms: Optional[Callable]=None,
        download: bool=False,
        checksum: bool=False,
        train_filename: str='bigearthnet-train.csv',
        val_filename: str='bigearthnet-val.csv',
        test_filename: str='bigearthnet-test.csv',
        keep_files_opened: bool=False
    ) -> None:
        super().__init__()
        self.root = Path(root)
        assert split in {'train', 'val', 'test'}
        self.split = split
        assert bands in {'s2', 's1', 'all'}
        if bands != 's2':
            raise ValueError(f'Not implemented yet {bands}')
        self.bands = bands
        assert num_classes in {19, 43}
        self.num_classes = num_classes
        self.transforms = transforms
        assert not download, f'Download must be False'
        assert not checksum, f'Checksum must be False'

        if split == 'train':
            self.filename = train_filename
        elif split == 'val':
            self.filename = val_filename
        else:
            self.filename = test_filename

        self.keep_files_opened = keep_files_opened

        self._load_metadata()
        self._load_hdf5()
        return
    
    def _load_metadata(self) -> None:
        self.df = pd.read_csv(self.root / self.filename)
        return
    
    def _load_hdf5(self) -> None:
        distinct_files = self.df['s2_hdf5_file'].unique()
        if self.keep_files_opened:
            self.hdf5 = {
                k: h5py.File(self.root / k, 'r') for k in distinct_files
            }
        else:
            self.hdf5 = {
                k: self.root / k for k in distinct_files
            }
        return
    
    def _get_from_hdf5(self, hdf5_filename: str, offset: int) -> Dict[str, np.ndarray]:
        value = self.hdf5[hdf5_filename]
        if isinstance(value, (str, Path)):
            # open file
            value = h5py.File(self.root / value, 'r')

        img = value['images'][offset]
        lbl = value[f'labels{self.num_classes}'][offset]
        res = {
            'image': img,
            'label': lbl
        }
        if not self.keep_files_opened:
            value.close()

        if not self.transforms is None:
            res = self.transforms(res)

        return res
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        hdf5_filename = row['s2_hdf5_file']
        offset = row['index']
        return self._get_from_hdf5(hdf5_filename, offset)
    
    def __len__(self) -> int:
        return self.df.shape[0]