from typing import List
import torch
from torch import nn
from torchinfo import summary
import torchvision
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.datasets import CocoDetection
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.model = nn.Sequential(
        # )
    
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
        # x = self.fc2(x)
        x = torch.relu(self.fc2(x))
        return x


class ResNet(nn.Module):
    def __init__(self, num=152):
        super().__init__()
        if num == 152:
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    def forward(self, x):
        return x


class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.model(x)
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        # class_num = num_classes + 1
        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_num)

    def forward(self, x):
        return self.model(x)

class Detector(nn.Module):
    def __init__(self, num_classes: int= 81, box_ratio: List[List[float]]=None):
        super().__init__()
        self.num_classes = num_classes
        if box_ratio == None:
            self.box_ratio = [[1.0,1.0], [1.5, 1.5], [1.0,2.0], [1.0,3.0], [2.0,1.0], [3.0,1.0]]
        else:
            self.box_ratio = box_ratio
    
    def forward(self, x):
        # x : [b,c,h,w]
        B,C,H,W = x.shape
        anker = nn.Conv2d(C, len(self.box_ratio) * (self.num_classes + 4), kernel_size=1, stride=1, padding=0)
        out = anker(x)
        return out

class SSD(nn.Module):
    def __init__(self, num_classes: int= 81, box_ratio: List[List[float]]=None):
        super().__init__()
        self.num_classes = num_classes
        if box_ratio == None:
            self.box_ratio = [[1.0,1.0], [1.5, 1.5], [1.0,2.0], [1.0,3.0], [2.0,1.0], [3.0,1.0]]
        else:
            self.box_ratio = box_ratio
        
        self.backbone = SwinTransformer()
        self.detector = Detector(num_classes=num_classes, box_ratio=box_ratio)
    
    def forward(self, x):
        assert x.shape[-1] >= 224 and x.shape[-2] >= 224
        features = self.backbone(x)
        out = self.detector(features)
        return out

class Conv(nn.Module):
    def __init__(self, input_size: List[int]=[675,675], num_classes: int= 81, box_ratio: List[List[float]]=None):
        super().__init__()
        self.num_classes = num_classes
        if box_ratio == None:
            self.box_ratio = [[1.0,1.0], [1.5, 1.5], [1.0,2.0], [1.0,3.0], [2.0,1.0], [3.0,1.0]]
        else:
            self.box_ratio = box_ratio
        self.input_size_h = input_size[0]
        self.input_size_w = input_size[1]
        # self.num_boxes = len(self.box_ratio)
        # swin_b = models.swin_b(weights='DEFAULT')
        # swin_b_modules = list(swin_b.children())[:-2]
        # self.backbone = nn.Sequential(*swin_b_modules)
        self.backbone = SwinTransformer()

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)

        self.anker1 = nn.Conv2d(512, len(self.box_ratio) * (self.num_classes + 4), kernel_size=3, stride=1, padding=0)
        self.anker2 = nn.Conv2d(256, len(self.box_ratio) * (self.num_classes + 4), kernel_size=3, stride=1, padding=0)
        self.anker3 = nn.Conv2d(256, len(self.box_ratio) * (self.num_classes + 4), kernel_size=1, stride=1, padding=0)
    
    def arrange_out(self, out):
        _,_,H,W = out.shape
        base_size_w = self.input_size_w / (W + 1)
        base_size_h = self.input_size_h / (H + 1)

        base_tensor = torch.arange(0, W).unsqueeze(1)
        base_tensor = base_tensor.repeat(1, H)
        x_tensor = base_tensor.t() * base_size_w

        base_tensor = torch.arange(0, H).unsqueeze(1)
        base_tensor = base_tensor.repeat(1, W)
        y_tensor = base_tensor * base_size_h

        for i in range(len(self.box_ratio)):
            out[:,(self.num_classes + 4) * i + 0] += x_tensor
            out[:,(self.num_classes + 4) * i + 1] += y_tensor
            out[:,(self.num_classes + 4) * i + 2] *= base_size_w
            out[:,(self.num_classes + 4) * i + 2] += self.box_ratio[i][0] * base_size_w
            out[:,(self.num_classes + 4) * i + 3] *= base_size_h
            out[:,(self.num_classes + 4) * i + 3] += self.box_ratio[i][1] * base_size_h

        # # [B,C,H,W] -> [B,H,W,C]
        out = out.permute(0, 2, 3, 1)
        # # [B,H,W,C] -> [B,H,W,len(self.box_ratio), self.num_classes + 4]
        out = out.view(out.size(0), out.size(1), out.size(2), len(self.box_ratio), self.num_classes + 4)
        return out
    
    def forward(self, x):
        out_list = []
        # input -> [B, 21, 21, 1024]
        features = self.backbone(x)
        # print(features.shape)

        # [B, 21, 21, 1024] -> [B, 21, 21, 512]
        features = torch.relu(self.conv1(features))
        out = self.anker1(features)
        # B,C,H,W = out.shape
        out = self.arrange_out(out)
        out_list.append(out)
        # print(out.shape)
        # print(out[:,:,:,:,4].shape)

        # [B, 21, 21, 512] -> [B, 19, 19, 1024]
        features = torch.relu(self.conv2(features))
        # [B, 19, 19, 1024] -> [B, 19, 19, 512]
        features = torch.relu(self.conv1(features))
        out = self.anker1(features)
        out = self.arrange_out(out)
        out_list.append(out)
        # print(out.shape)

        # [B, 19, 19, 512] -> [B, 19, 19, 256]
        features = torch.relu(self.conv3(features))
        # [B, 19, 19, 256] -> [B, 10, 10, 512]
        features = torch.relu(self.conv4(features))
        out = self.anker1(features)
        out = self.arrange_out(out)
        out_list.append(out)
        # print(out.shape)

        # [B, 10, 10, 512] -> [B, 10, 10, 128]
        features = torch.relu(self.conv5(features))
        # [B, 10, 10, 128] -> [B, 5, 5, 256]
        features = torch.relu(self.conv6(features))
        out = self.anker2(features)
        out = self.arrange_out(out)
        out_list.append(out)
        # print(out.shape)

        # [B, 5, 5, 256] -> [B, 5, 5, 128]
        features = torch.relu(self.conv7(features))
        # [B, 5, 5, 128] -> [B, 3, 3, 256]
        features = torch.relu(self.conv8(features))
        out = self.anker2(features)
        out = self.arrange_out(out)
        out_list.append(out)
        # print(out.shape)

        # [B, 3, 3, 256] -> [B, 3, 3, 128]
        features = torch.relu(self.conv7(features))
        # [B, 3, 3, 128] -> [B, 1, 1, 256]
        features = torch.relu(self.conv8(features))
        out = self.anker3(features)
        out = self.arrange_out(out)
        out_list.append(out)

        return out_list

if __name__ == "__main__":
    size = 675
    # size = 224

    # model = MyModel()
    model = SwinTransformer()
    # model = models.swin_b(weights='DEFAULT')
    # model = ResNet(152)
    print(summary(model, input_size=(1, 3, size, size), device=torch.device("cpu")))
    # model = FasterRCNN()
    boxes = [[1.0,1.0], [1.5, 1.5], [1.0,2.0], [1.0,3], [2.0,1.0], [3.0,1.0]]
    model = Conv(num_classes=81, box_ratio=boxes, input_size=[size, size])
    # model.eval()
    # print(model)
    # print(summary(model, input_size=(1, 3, size, size), device=torch.device("cpu")))
    model.eval()

    # fake image
    # size = 224
    # [b,c,h,w]
    img = torch.randn(1, 3, size, size*2)
    # preprocess = models.Swin_B_Weights.IMAGENET1K_V1.transforms()
    # img = preprocess(img)
    output = model(img)
    # print(output.shape)
    # print(size)
    # print(output[1].shape)
    for out in output:
        print(out.shape)
        # print((type(out[1])))
    # in 644~675  out 21

    # _out = output[0]
    # print(_out.shape)
    # print(_out[:,:,:,:,0].shape)
    # print(_out[:,0].shape)
    # new_out = []
    # for out in output:
    #     print(out.shape)
    #     B,H,W,_box,_class = out.shape
    #     # if H != 1:
    #     #     H += 1
    #     # if W != 1:
    #     #     W += 1
    #     base_x = size / (W + 1)
    #     base_y = size / (H + 1)
    #     for i in range(H):
    #         for j in range(W):
    #             if i == 2 and j == 2:
    #                 print(out[:,i,j,2,0])
    #             out[:,i,j,:,0] = out[:,i,j,:,0] * base_x + base_x * j + base_y * i
    #             out[:,i,j,:,1] = out[:,i,j,:,1] * base_y + base_x * j + base_y * i
    #             if i == 2 and j == 2:
    #                 print(out[:,i,j,2,0])
    #                 print("")
    #                 print("")
        # new_out.append(out)
    # new_out = np.array(new_out)
    # new_out = torch.from_numpy(new_out)
    # print(output[3][:,2,2,2,0])
    # print()
    # # print(new_out[3][:,2,2,2,0])
    # print()
    # print(output.shape)
    # print(new_out.shape)
    # print()
    # print((new_out[0] - output[0])[:,2,2,2,:4])

    """
    # real image
    dataset = CocoDetection("../data/coco/train2017/", "../data/coco/annotations/instances_train2017.json")
    sample1 = dataset[8]
    img1, target1 = sample1
    sample2 = dataset[5]
    img2, target2 = sample2

    preprocess = transforms.Compose([
        transforms.ToTensor(),               # PIL画像をPyTorchテンソルに変換
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNetの平均と標準偏差で正規化
    ])
    preprocess = models.Swin_B_Weights.IMAGENET1K_V1.transforms()

    img1_tensor = preprocess(img1)
    img2_tensor = preprocess(img2)

    # input_batch = torch.stack([img1, img2], dim=0)
    input_batch = img1_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

        boxes = output[0]["boxes"].data
        scores = output[0]["scores"].data
        labels = output[0]["labels"].data

        boxes = boxes[scores >= 0.5]
        scores = scores[scores >= 0.5]

        for i, box in enumerate(boxes):
            draw = ImageDraw.Draw(img1)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

        img1.save(f"faster_rcnn_test.png")

    print(output)
    print(output.shape)
    """
