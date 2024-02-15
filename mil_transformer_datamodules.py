from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import datasets


class WsiMilDataModule(LightningDataModule):
    name = "WsiMilDataModule"

    def __init__(
        self,
        dataset: str = "./TCGA",
        test_features_dir: str = "",
        bag_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 8,
        target: str = "ER",
        val_fold: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bag_size = bag_size
        self.val_fold = val_fold

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset_train = datasets.WSI_MILdataset(
                DataSet_location=self.dataset,
                bag_size=self.bag_size,
                target_kind=self.target,
                test_fold=self.val_fold,
                train=True,
                transform_type="pcbnfrsc",
            )

            self.dataset_val = datasets.WSI_MILdataset(
                DataSet_location=self.dataset,
                bag_size=self.bag_size,
                target_kind=self.target,
                test_fold=self.val_fold,
                train=False,
                transform_type="none",
            )
        elif stage == "test":
            self.dataset_test = datasets.WSI_MILdataset(
                DataSet_location=self.dataset,
                bag_size=self.bag_size,
                target_kind=self.target,
                test_fold=self.val_fold,
                train=False,
                transform_type="none",
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class WsiMilFeaturesDataModule(LightningDataModule):
    name = "WsiMilFeaturesDataModule"

    def __init__(
        self,
        dataset: str = "./TCGA",
        train_features_dir: str = "",
        data_location: dict = None,
        test_features_dir: str = "",
        test_dataset: str = "",
        bag_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 8,
        target: str = "ER",
        batch: int = 9,
        magnification: int = 10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_location = data_location
        try:
            self.train_features_dir = (
                train_features_dir
                if train_features_dir != ""
                else data_location["TrainSet Location"]
            )
        except KeyError:
            self.train_features_dir = ""
            print(
                "Could not find train features dir data location in dict, ignore this if running inference only"
            )
        self.val_features_dir = data_location["TestSet Location"]
        self.test_features_dir = (
            test_features_dir
            if test_features_dir != ""
            else data_location["TestSet Location"]
        )
        self.dataset = dataset
        self.test_dataset = test_dataset if test_dataset != "" else self.dataset
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bag_size = bag_size
        self.magnification = magnification

    def setup(self, stage=None):
        if stage == "fit":
            self.dataset_train = datasets.Features_MILdataset(
                dataset_location=self.dataset,
                data_location=self.train_features_dir,
                bag_size=self.bag_size,
                target=self.target,
                is_train=True,
                slide_magnification = self.magnification,
            )

            self.dataset_val = datasets.Features_MILdataset(
                data_location=self.val_features_dir,
                dataset_location=self.dataset,
                bag_size=self.bag_size,
                target=self.target,
                minimum_tiles_in_slide=self.bag_size,
                is_train=False,
                slide_magnification = self.magnification,
            )
            
        elif stage == "test":
            self.dataset_test = datasets.Features_MILdataset(
                data_location=self.test_features_dir,
                dataset_location=self.test_dataset,
                target=self.target,
                is_all_tiles=True,
                minimum_tiles_in_slide=self.bag_size,
                is_train=False,
                slide_magnification = self.magnification,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
