import os
import sys
import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO


class Visualize():

    def vis_dataset(self, data_loader=None, model=None, classes= None, save_path="results.png", mean=None, std=None, max_in_row=5, skip=0, to_wandb=False):
        class_num = len(classes)
        image_list = []
        class_list = []

        result_image_list = []
        result_gt_list = []
        result_pred_list = []

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
            denormalized_image = denormalized_image.numpy()

            if model != None:
                with torch.no_grad():
                    pred = model(image.unsqueeze(0).to('cuda'))
                pred = pred.argmax().item()

                result_image_list.append(denormalized_image)
                result_gt_list.append(label)
                result_pred_list.append(classes[pred])

                label = f"GT: {label}\nPred: {classes[pred]}"

            
            axs[row, col].imshow(denormalized_image)
            axs[row, col].set_title(label)
            axs[row, col].axis('off')

        plt.tight_layout()
        if to_wandb == True and model != None:
            columns = ["image", "Ground Truth", "Prediction"]
            data = []
            for i in range(class_num):
                data.append([
                    wandb.Image(result_image_list[i]), result_gt_list[i], result_pred_list[i]
                ])
            test_table = wandb.Table(
                columns=columns, data=data
            )
            wandb.log({"final_predictions" : test_table})
            print("send to wandb")
        
        if "/" in save_path:
            _dir = ""
            c = save_path.split("/")
            for i in range(len(c) - 1):
                if i != 0:
                    _dir += "/"
                _dir += c[i]
            if not os.path.exists(_dir):
                print(f"Directory not found: {_dir}")
                print(f"Make directory : {_dir}")
                os.makedirs(_dir)
        plt.savefig(save_path)
        plt.close()
    
    def coco_vis(self, img_id:int=64, draw_bbox=True):

        anno_path = "../data/coco/annotations/instances_train2017.json"
        if not os.path.exists(anno_path):
            print(f"file not found: {anno_path}")
            sys.exit()
        coco = COCO(anno_path)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annos = coco.loadAnns(ann_ids)

        img = coco.loadImgs(img_id)[0]
        image_path = '../data/coco/train2017/' + img['file_name']
        if not os.path.exists(anno_path):
            print(f"file not found: {image_path}")
            sys.exit()
        I = plt.imread(image_path)

        plt.imshow(I); 
        coco.showAnns(anns=annos, draw_bbox=draw_bbox)
        plt.axis('off')

        plt.savefig(f'result_coco_vis_{img_id}.png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    vis_func = Visualize()
    vis_func.coco_vis()
