import os
from typing import List, Dict, Union
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import CocoDetection
from torchvision import transforms
import lightning as pl


class MyDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            data_dir: str = "./", 
            seed:int = 42, 
            batch_size: int = 64, 
            workers_num: int = 1, 
            train_val_ratio: float = 0.8, 
            transform_dic: Dict[str, List[float]] = None
        ):
        super().__init__()
        self.save_hyperparameters()
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.gene = torch.Generator().manual_seed(seed)
        if transform_dic != None:
            self.mean = self.hparams.transform_dic['mean']
            self.std = self.hparams.transform_dic['std']
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        self.transform = self.transform = transforms.Compose([transforms.ToTensor(),  transforms.Normalize(self.mean, self.std)])
    
    def print_hyperparameters(self):
        print(self.hparams)
    
    def get_mean_std(self):
        return self.mean, self.std

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = CocoDetection(root=os.path.join(self.hparams.data_dir, "train2017/"), annFile=os.path.join(self.hparams.data_dir, "annotations/instances_train2017.json"), transform=self.transform)
            self.val_data = CocoDetection(root=os.path.join(self.hparams.data_dir, "val2017/"), annFile=os.path.join(self.hparams.data_dir, "annotations/instances_val2017.json"),transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.val_data = CocoDetection(root=os.path.join(self.hparams.data_dir, "val2017/"), annFile=os.path.join(self.hparams.data_dir, "annotations/instances_val2017.json"),transform=self.transform)

        if stage == "predict":
            self.val_data = CocoDetection(root=os.path.join(self.hparams.data_dir, "val2017/"), annFile=os.path.join(self.hparams.data_dir, "annotations/instances_val2017.json"),transform=self.transform)

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


class CustomCOCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.coco = CocoDetection(root=root, annFile=annFile)
        self.transform = transform

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, idx):
        img, target = self.coco[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

if __name__ == "__main__":
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # dic = {"mean": mean, "std": std}
    # dataloader = MyDataModule(transform_dic=dic)
    # dataloader._print_hyperparameters()

    dataset = CocoDetection("../data/coco/train2017/", "../data/coco/annotations/instances_train2017.json")
    # coco = COCO("../data/coco/annotations/instances_train2017.json")
    """
    w_min = 10000
    w_max = 0
    h_min = 10000
    h_max = 0
    count1 = 0
    count2 = 0
    count3 = 0

    for sample in dataset:
        img, target = sample
        w,h = img.size
        if w > w_max:
            w_max = w
        if w < w_min:
            w_min = w
        if h > h_max:
            h_max = h
        if h < h_min:
            h_min = h

        if w < h:
            count1 += 1
        elif w == h:
            count2 += 1
        elif w > h:
            count3 += 1
    
    print(f"w_min: {w_min}")
    print(f"w_max: {w_max}")
    print(f"h_min: {h_min}")
    print(f"h_max: {h_max}")
    print(f"h>w: {count1}\nh==w: {count2}\nh<w: {count3}")
    # w_min: 59
    # w_max: 640
    # h_min: 51
    # h_max: 640
    # h>w: 27948
    # h==w: 4324
    # h<w: 86015

    max_boxes = 0
    for sample in dataset:
        img, target = sample
        if len(target) > max_boxes:
            max_boxes = len(target)
    
    print(f"max_boxes: {max_boxes}")
    # max 93
    """

    sample = dataset[8]
    img, target = sample
    print(str(target[0]))
    print(f"image_id : {target[0]['image_id']}")
    # img.save("./image.png")
    # plt.imshow(img)
    # plt.axis('off')

    # for t in target:
    #     print(t['bbox'])
    #     print(t['image_id'])
    #     print(t['category_id'])
    #     print(img.size) # (w, h)
    #     bbox = t['bbox']
    #     class_name = coco.loadCats(t['category_id'])[0]['name']
    #     plt.plot([bbox[0], bbox[0] + bbox[2], bbox[0] + bbox[2], bbox[0], bbox[0]], 
    #          [bbox[1], bbox[1], bbox[1] + bbox[3], bbox[1] + bbox[3], bbox[1]], 
    #          color='red', linewidth=2
    #     )
    #     plt.text(bbox[0], bbox[1], class_name, fontsize=12, color='red')
    #     print()
    print(f"type(img) = {type(img)}\ntype(target) = {type(target)}\ntype(target[0]) = {type(target[0])}\ntarget[0].keys() = {target[0].keys()}")
    # plt.savefig('./image_with_bbox.png')

    # data_dir = "../data/coco"

    # print(os.path.join(data_dir, "val2017/"))
    # print(os.path.join(data_dir, "annotations/instances_val2017.json"))
