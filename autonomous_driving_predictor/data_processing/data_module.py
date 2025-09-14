from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from .dataset import MultiDataset
from autonomous_driving_predictor.transformation_utils.target_builder import WaymoTargetBuilder


class DataModule(pl.LightningDataModule):
    transforms = {
        "WaymoTargetBuilder": WaymoTargetBuilder,
    }

    dataset = {
        "scalable": MultiDataset,
    }

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 transform: Optional[str] = None,
                 dataset: Optional[str] = None,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 processor='ntp',
                 use_intention=False,
                 token_size=512,
                 **kwargs) -> None:
        super(DataModule, self).__init__()
        self.root = root
        self.dataset_class = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.processor = processor
        self.use_intention = use_intention
        self.token_size = token_size

        train_transform = DataModule.transforms[transform](num_historical_steps, num_future_steps, "train")
        val_transform = DataModule.transforms[transform](num_historical_steps, num_future_steps, "val")
        test_transform = DataModule.transforms[transform](num_historical_steps, num_future_steps)

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: Optional[str] = None) -> None:
        # Data is already split in waymo_open_dataset/{train,val,test} directories
        self.train_dataset = DataModule.dataset[self.dataset_class](self.root, 'train', processor=self.processor, transform=self.train_transform, token_size=self.token_size)
        self.val_dataset = DataModule.dataset[self.dataset_class](self.root, 'val', processor=self.processor, transform=self.val_transform, token_size=self.token_size)
        self.test_dataset = DataModule.dataset[self.dataset_class](self.root, 'test', processor=self.processor, transform=self.test_transform, token_size=self.token_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
