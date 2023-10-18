import os
from typing import List
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping


class Config:
    data_dir: str = "../data/"
    save_dir: str = "logs/"
    batch_size: int = 256 if torch.cuda.is_available() else 64
    max_epochs: int = 300
    learning_rate: float = 0.001
    accelerator: str = "gpu"
    devices: List[int] = [0, 2, 4, 6]
    max_model_num: int = 1
    workers_num: int = 16 # os.cpu_count()
    seed: int = 42
    train_val_ratio: float = 0.8
    early_stopping: bool = False
    early_stopping_patience: int = 50

config = Config()

def vis_dataset(data_loader=None, model=None, classes= None, save_path="cifar10_sample_images.png", mean=None, std=None, max_in_row=5, skip=0):
    # classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_num = len(classes)
    image_list = []
    class_list = []

    dataiter = iter(data_loader)
    for i in range(skip):
        next(dataiter)
    while len(class_list) != class_num:
        images, labels = next(dataiter)
        for i,img in enumerate(images):
            if labels[i] not in class_list:
                class_list.append(labels[i])
                image_list.append(img)
            if len(class_list) == class_num:
                break
    
    _row = class_num // max_in_row
    if class_num % max_in_row != 0:
        _row += 1
    fig, axs = plt.subplots(_row, max_in_row)
    for i in range(class_num):
        idx = class_list.index(i)
        image = image_list[idx]
        label = classes[i]

        row = i // max_in_row
        col = i % max_in_row

        # std = (100, 0, 1)
        denormalized_image = image * std[:, np.newaxis, np.newaxis]
        denormalized_image += mean[:, np.newaxis, np.newaxis]
        denormalized_image = (denormalized_image * 255).to(torch.uint8)
        # (c, h, w) -> (h, w, c)
        denormalized_image = denormalized_image.permute(1, 2, 0)
        # print(f"{denormalized_image}\n")

        if model != None:
            with torch.no_grad():
                pred = model(image.unsqueeze(0).to('cuda'))
            pred = pred.argmax().item()
            label = f"GT:{label}\nP:{classes[pred]}"
        
        axs[row, col].imshow(denormalized_image)
        axs[row, col].set_title(label)
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # print("----------------------- fin vis -----------------------")


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
        print("Data mean", self.mean)
        print("Data std", self.std)
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


class MyNet(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # プーリング層の追加
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _shared_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(1) == labels).type(torch.float)
        # acc = (outputs.argmax(1) == labels).type(torch.float).mean()
        # acc = (outputs.argmax(dim=-1) == labels).float().mean()
        return loss, acc
    
    def _shared_log(self, stage=None, prog: bool=False):
        loss = -1
        accuracy = -1
        if stage == "train":
            loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
            accuracy = torch.cat([x['accuracy'] for x in self.train_step_outputs]).mean()
        elif stage == "val":
            loss = torch.stack([x['val_loss'] for x in self.val_step_outputs]).mean()
            accuracy = torch.cat([x['val_accuracy'] for x in self.val_step_outputs]).mean()
        elif stage == "test":
            loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
            accuracy = torch.cat([x['test_accuracy'] for x in self.test_step_outputs]).mean()
        else:
            return
        self.log(
            f"{stage}_loss", #metiricsの名前
            loss,#metiricsの値
            prog_bar=prog,#プログレスバーに表示するか？
            # logger=True,#結果を保存するのか？
            # on_epoch=True,#１epoch中の結果を累積した値を利用するのか？
            # on_step=True,#１stepの結果を利用するのか？
            sync_dist=True
            # rank_zero_only=True
        )
        self.log(
            f"{stage}_accuracy", #metiricsの名前
            accuracy,#metiricsの値
            prog_bar=prog,#プログレスバーに表示するか？
            # logger=True,#結果を保存するのか？
            # on_epoch=True,#１epoch中の結果を累積した値を利用するのか？
            # on_step=True,#１stepの結果を利用するのか？
            sync_dist=True
            # rank_zero_only=True
        )
        return

    # train
    # def on_train_start(self):
    #     pass

    # def on_train_epoch_start(self):
    #     pass

    # def on_train_batch_start(self, batch, batch_idx):
    #     pass

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)
        self.train_step_outputs.append({"loss": loss, "accuracy": acc})
        self.log(
            "train_step_loss", 
            loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True
        )
        self.log(
            "train_step_accuracy", 
            acc.mean(),
            on_epoch=True,
            on_step=True,
            sync_dist=True
        )
        return {"loss": loss, "accuracy": acc}
    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     pass

    def on_train_epoch_end(self):
        # self._shared_log(stage="train")
        self.train_step_outputs.clear()
        return

    # def on_train_end(self):
    #     pass

    # validation
    # def on_validation_start(self):
    #     pass

    # def on_validation_epoch_start(self):
    #     pass

    # def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     pass
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)
        self.val_step_outputs.append({"val_loss": loss, "val_accuracy": acc})
        return {"val_loss": loss, "val_accuracy": acc}
    
    # def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     pass
    
    def on_validation_epoch_end(self) -> None:
        self._shared_log(stage="val", prog=True)
        self.val_step_outputs.clear()
        return
    
    # def on_validation_end(self):
    #     pass

    #test
    # def on_test_start(self):
    #     pass

    # def on_test_epoch_start(self):
    #     pass
    
    # def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
    #     pass

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)
        self.test_step_outputs.append({"test_loss": loss, "test_accuracy": acc})
        return {"test_loss": loss, "test_accuracy": acc}
    
    # def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     pass

    def on_test_epoch_end(self) -> None:
        self._shared_log(stage="test")
        self.test_step_outputs.clear()
        return

    # def on_test_end(self):
    #     pass

    # predict
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        return outputs
        # return {"preds": outputs.argmax(1), "gts": labels}


if __name__ == "__main__":
    # check
    # print(f"cpu core : {os.cpu_count()}")
    # print(f"is {torch.cuda.is_available()}")
    # print(f"torch version : {torch.__version__}")
    # print(f"cuda version : {torch.version.cuda}")
    # print(f"CuDNN version : {torch.backends.cudnn.version()}")
    # print(f"is CuDNN enable : {torch.backends.cudnn.enabled}")
    # print(f"NCCL version : {torch.cuda.nccl.version()}")

    pl.pytorch.seed_everything(config.seed, workers=True)

    datamodule = MyDataModule(
        data_dir=config.data_dir, 
        seed=config.seed, 
        batch_size=config.batch_size, 
        workers_num=config.workers_num, 
        train_val_ratio=config.train_val_ratio
    )
    model = MyNet(config.learning_rate)
    callbacks = []

    if config.early_stopping:
        callbacks.append(
            EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience)
        )
    
    # save top k models
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./pl-checkpoints",
        filename="cifar10_classification",
        save_top_k=1,
        mode="min",
    )
    callbacks.append(checkpoint_callback)
    
    # trainer = pl.Trainer(max_epochs=config.max_epochs, log_every_n_steps=1, accelerator=config.accelerator, devices=config.devices, callbacks=callbacks, deterministic=True)
    trainer = pl.Trainer(max_epochs=config.max_epochs, log_every_n_steps=1, accelerator=config.accelerator, devices=config.devices, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)

    # trainer = pl.Trainer(devices=[0], num_nodes=1)
    trainer.test(model, datamodule=datamodule)


    try:
        mean, std = datamodule.get_mean_std()
    except:
        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        datamodule.setup(stage="test")
        mean, std = datamodule.get_mean_std()
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    vis_dataset(data_loader=datamodule.train_dataloader(), classes=classes, save_path="cifar10_sample_images.png", mean=mean, std=std, skip=10)

    model = MyNet.load_from_checkpoint(
        checkpoint_path="./pl-checkpoints/cifar10_classification.ckpt"
    )
    model.to('cuda')

    vis_dataset(data_loader=datamodule.test_dataloader(), model=model, classes=classes, save_path="cifar10_predict_images.png", mean=mean, std=std, skip=10)
