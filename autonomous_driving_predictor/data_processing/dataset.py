import os
import pickle
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd
from torch_geometric.data import Dataset
from autonomous_driving_predictor.core_algorithms.log import Logging
import numpy as np
from .data_preprocessor import TokenProcessor


def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


class MultiDataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: List[str] = None,
                 processed_dir: List[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True,
                 cluster: bool = False,
                 processor=None,
                 use_intention=False,
                 token_size=512) -> None:
        self.logger = Logging().log(level='DEBUG')
        self.root = root
        self.well_done = [0]
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split
        self.training = split == 'train'
        self.logger.debug("Starting loading dataset")
        self._raw_file_names = []
        self._raw_paths = []
        self._raw_file_dataset = []
        if raw_dir is not None:
            self._raw_dir = raw_dir
            for raw_dir in self._raw_dir:
                raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
                dataset = "waymo"
                file_list = os.listdir(raw_dir)
                self._raw_file_names.extend(file_list)
                self._raw_paths.extend([os.path.join(raw_dir, f) for f in file_list])
                self._raw_file_dataset.extend([dataset for _ in range(len(file_list))])
        if self.root is not None:
            # Data is already split - scan the appropriate directory
            # Map 'val' to 'validation' to match directory naming
            split_name = 'validation' if split == 'val' else split
            split_dir = os.path.join(root, "waymo_open_dataset", split_name)
            if os.path.exists(split_dir):
                file_list = [f for f in os.listdir(split_dir) if f.endswith('.pkl')]
                self._processed_file_names = file_list
                self._processed_paths = [os.path.join(split_dir, f) for f in file_list]
            else:
                self.logger.warning(f"Split directory {split_dir} not found, using empty list")
                self._processed_file_names = []
                self._processed_paths = []
        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self._num_samples = len(self._processed_file_names) - 1 if processed_dir is not None else len(self._raw_file_names)
        self.logger.debug("The number of {} dataset is ".format(split) + str(self._num_samples))
        self.token_processor = TokenProcessor(2048)
        super(MultiDataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def len(self) -> int:
        return self._num_samples

    def generate_ref_token(self):
        pass

    def get(self, idx: int):
        # Use processed paths if available, otherwise fall back to raw paths
        if hasattr(self, '_processed_paths') and self._processed_paths:
            file_path = self._processed_paths[idx]
        else:
            file_path = self.raw_paths[idx]
        
        with open(file_path, 'rb') as handle:
            data = pickle.load(handle)
        data = self.token_processor.preprocess(data)
        return data
