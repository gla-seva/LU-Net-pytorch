# import torch
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split

os.chdir("/home/vglasov/Reseach/LU-Net-pytorch/")
import config

from tools import preprocess #custom class
from tools import dataloader_tools as data_loader
import open3d as o3d

import torch
from torch.utils.data import DataLoader, Dataset

class batch_loader(Dataset):
    def __init__(self, root_dir, augmentation=None):
        self.landmarks_frame = pd.read_csv(root_dir)
        self.augmentation = augmentation
        
    def _read_labels(self, labels_path):
        return [list(map(float, f.split()[4:10])) for f in open(labels_path, "r").readlines()]

    def __getitem__(self, idx):
        pcd_path = self.landmarks_frame.iloc[idx, 0]
        labels_path = self.landmarks_frame.iloc[idx, 1]

        pcd = o3d.io.read_point_cloud(pcd_path)
        labels_list = self._read_labels(labels_path)
        
        pcd2img = preprocess.Pcd2ImageTransform(augmentation=self.augmentation).fit(pcd, labels_list)
        data = pcd2img.transform()
        
        mask = data[:,:,0] != 0
        p, n = data_loader.pointnetize(data[:,:,0:4], n_size=[3,3])
        groundtruth = data_loader.apply_mask(data[:,:,-1], mask)
        
        return torch.tensor(p, dtype=torch.float).permute(-1, -2, 0, 1),\
               torch.tensor(n, dtype=torch.float).permute(-1, -2, 0, 1),\
               torch.tensor(mask),\
               torch.tensor(groundtruth)
    
    def __len__(self):
        return (len(self.landmarks_frame))
    
# if __name__ == '__main__':
#     train_loader = batch_loader(root_dir="data/train.csv")
#     val_loader = batch_loader(root_dir="data/test.csv")