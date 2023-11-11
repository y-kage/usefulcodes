import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import lightning as pl


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", seed:int = 42, batch_size: int = 64, workers_num: int = 1, train_val_ratio: float = 0.8):
        super().__init__()
        self.save_hyperparameters()
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.gene = torch.Generator().manual_seed(seed)

    def prepare_data(self):
        # download
        torchvision.datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # if not (self.transform != None and self.mean != None and self.std != None):
        train_dataset = torchvision.datasets.CIFAR10(self.hparams.data_dir, train=True)
        self.mean = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
        self.std = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
        # print("Data mean", self.mean)
        # print("Data std", self.std)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # add some augmentation
            train_transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
            full_data = torchvision.datasets.CIFAR10(self.hparams.data_dir, train=True, transform=train_transform)
            train_set_size = int(len(full_data) * self.hparams.train_val_ratio)
            val_set_size = len(full_data) - train_set_size
            self.train_data, self.val_data = torch.utils.data.random_split(full_data, [train_set_size, val_set_size], generator=self.gene)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = torchvision.datasets.CIFAR10(self.hparams.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.predict_data = torchvision.datasets.CIFAR10(self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.workers_num)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.workers_num)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.workers_num)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.workers_num)
    
    def teardown(self, stage):
        pass

    def get_mean_std(self):
        return self.mean, self.std


if __name__ == "__main__":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    dic = {"mean": mean, "std": std}
    dataloader = MyDataModule(transform_dic=dic)
    dataloader._print_hyperparameters()
