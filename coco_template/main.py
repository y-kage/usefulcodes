import os
import sys
import yaml
import argparse
from typing import List
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import wandb
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import lightning as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# import classes from local files
# from model import MyModel as mModel
# from model import SwinTransformer as swinT_b
# from model import FasterRCNN as mModel
from model import SSD as mModel
from dataloader import MyDataModule as mLoader
from loss_func import Loss_func
from tools import Visualize as vis


def get_args():
    parser = argparse.ArgumentParser(description='template of pytorch lightning.')

    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--accelerator', type=str, default="gpu")
    parser.add_argument('--devices', default=-1)
    parser.add_argument('--workers_num', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_val_ratio', type=float, default=0.8)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=50)
    parser.add_argument('--logger', default=None, choices=[None, "wandb"])
    parser.add_argument('--mode', type=str, default="all", choices=["all", "train", "test"])

    return parser.parse_args()

class MyNet(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.001, num_classes=80, input_size = 675.0, default_boxes: List[List[float]]=None):
        super().__init__()
        # self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.num_classes = num_classes + 1 # add background
        self.input_size = input_size
        self.mymodel = mModel(num_classes=self.num_classes, box_ratio=default_boxes)
        self.loss_func = Loss_func(self.num_classes)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        return self.mymodel(x)
    
    def compute_loss(self, gt_bbox, gt_class, preds, alpha=1.0):
        pass
    
    def _shared_step(self, batch, batch_idx):
        # 675 -> 21 -> 19 -> 10 -> 5 -> 3 -> 1
        inputs, targets = batch
        # [B, gt_bboxes, 5]  5-> x,y,w,h,class
        gt_classes = targets[:][:][4]
        gt_classes += 1 # add background
        gt_bbox = targets[:][:][:4]
        # output[i] : [B, H, W, num_default_box, class_num + 4], 0 <= i <= 5
        outputs = self(inputs)

        # loc_pred, conf_pred = predictions
        # loc_targets, conf_targets = targets

        self.loss_func(outputs, targets)



        loss = 0

        gt_classes = targets[:][:][4]
        gt_bbox = targets[:][:][:4]


        return loss, acc
    
    def _shared_log(self, stage=None, outputs=None, prog: bool=False):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        accuracy = torch.cat([x["accuracy"] for x in outputs]).mean()

        self.log(
            f"{stage}_loss", # metirics name
            loss, # metirics variable
            prog_bar=prog,
            sync_dist=True
        )
        self.log(
            f"{stage}_accuracy",
            accuracy,
            prog_bar=prog,
            sync_dist=True
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
            "train_loss", 
            loss,
            on_epoch=True,
            on_step=True,
            sync_dist=True
        )
        self.log(
            "train_accuracy", 
            acc.mean(),
            on_epoch=True,
            on_step=True,
            sync_dist=True
        )
        return {"loss": loss, "accuracy": acc}
    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     pass

    # def on_train_epoch_end(self):
    #     self._shared_log(stage="train", outputs=self.train_step_outputs)
    #     self.train_step_outputs.clear()
    #     return

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
        self.val_step_outputs.append({"loss": loss, "accuracy": acc})
        return {"val_loss": loss, "val_accuracy": acc}
    
    # def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     pass
    
    def on_validation_epoch_end(self) -> None:
        self._shared_log(stage="val", outputs=self.val_step_outputs, prog=True)
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
        self.test_step_outputs.append({"loss": loss, "accuracy": acc})
        return {"test_loss": loss, "test_accuracy": acc}
    
    # def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     pass

    def on_test_epoch_end(self) -> None:
        self._shared_log(stage="test", outputs=self.test_step_outputs)
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

    # wandb_logger = WandbLogger(project="pl_template")

    args = get_args()
    # with open('config.yaml', 'r') as stream:
    #     try:
    #         data = yaml.safe_load(stream)
    #         print(data)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #         sys.exit()

    pl.pytorch.seed_everything(args.seed, workers=True)

    datamodule = mLoader(
        data_dir=args.data_dir, 
        seed=args.seed, 
        batch_size=args.batch_size, 
        workers_num=args.workers_num, 
        train_val_ratio=args.train_val_ratio
    )

    if args.mode == "all" or args.mode == "train":

        model = MyNet(args.learning_rate)
        print(model)
        callbacks = []

        if args.early_stopping:
            callbacks.append(
                EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience)
            )
        
        # save top k models
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="./pl-checkpoints",
            filename="cifar10_classification_{val_loss:.2f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            mode="min",
        )
        callbacks.append(checkpoint_callback)

        if args.logger == "wandb":
            args.logger = WandbLogger()
        
        # trainer = pl.Trainer(max_epochs=args.max_epochs, log_every_n_steps=1, accelerator=args.accelerator, devices=args.devices, callbacks=callbacks, deterministic=True)
        trainer = pl.Trainer(logger=args.logger, max_epochs=args.max_epochs, log_every_n_steps=1, accelerator=args.accelerator, devices=args.devices, callbacks=callbacks)
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)
    
    if args.mode == "all" or args.mode == "test":

        skip = 7
        to_wandb = False

        try:
            mean, std = datamodule.get_mean_std()
        except:
            datamodule.prepare_data()
            datamodule.setup(stage="fit")
            datamodule.setup(stage="test")
            mean, std = datamodule.get_mean_std()
        
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        vis.vis_dataset(data_loader=datamodule.train_dataloader(), classes=classes, save_path="./results/cifar10_sample_images.png", mean=mean, std=std, skip=skip)

        min_loss = 100
        max_checkpoint = None
        checkpoint_list = os.listdir("./pl-checkpoints")
        for c in checkpoint_list:
            _v = c.split("_")[-1]
            if not ".ckpt" in _v:
                print("No ckpt")
                sys.exit()
            _v = _v.split(".ckpt")[0]
            if float(_v) < min_loss:
                min_loss = float(_v)
                max_checkpoint = c
        print(f"Checkpoint : {max_checkpoint}")
        model = MyNet.load_from_checkpoint(
            checkpoint_path=f"./pl-checkpoints/{max_checkpoint}"
        )
        model.eval()
        model.to('cuda')

        if args.logger == "wandb":
            to_wandb = True

        vis.vis_dataset(data_loader=datamodule.test_dataloader(), model=model, classes=classes, save_path="./results/cifar10_predict_images.png", mean=mean, std=std, skip=skip, to_wandb=to_wandb)
