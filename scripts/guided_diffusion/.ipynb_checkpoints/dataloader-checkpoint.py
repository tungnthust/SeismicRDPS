import torch
import numpy as np
from math import ceil

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    image = image * 2 - 1
    return image

def split_patch(data, patch_size):
    depth, num_ilines, num_xlines = data.shape
    num_patch_width_ilines = ceil(num_xlines / patch_size)
    num_patch_width_xlines = ceil(num_ilines / patch_size)
    num_patch_depth = ceil(depth / patch_size)
    patches = []
    for iline in range(num_ilines):
        for y in range(num_patch_depth):
            if y == (num_patch_depth - 1):
                start_y = -patch_size
                end_y = depth
            else:
                start_y = y*patch_size
                end_y = (y+1)*patch_size
            for x in range(num_patch_width_ilines):
                if x == (num_patch_width_ilines - 1):
                    start_x = -patch_size
                    end_x = num_xlines
                else:
                    start_x = x*patch_size
                    end_x = (x+1)*patch_size
                patches.append(data[start_y:end_y, iline, start_x:end_x])
    for xline in range(num_xlines):
        for y in range(num_patch_depth):
            if y == (num_patch_depth - 1):
                start_y = -patch_size
                end_y = depth
            else:
                start_y = y*patch_size
                end_y = (y+1)*patch_size
            for x in range(num_patch_width_xlines):
                if x == (num_patch_width_xlines - 1):
                    start_x = -patch_size
                    end_x = num_ilines
                else:
                    start_x = x*patch_size
                    end_x = (x+1)*patch_size
                patches.append(data[start_y:end_y, start_x:end_x, xline])
    return patches
    
class SeismicDataset(torch.utils.data.Dataset):
    def __init__(self, directory='/poc-data/pvn/data', mode="train", datasets=['F3', 'Kerry3D'], patch_size=128, transform=None):
        
        super().__init__()
        self.data = []
        self.transform = transform
        for dataset in datasets:
            data = np.load(f'{directory}/{dataset}/{mode}.npy').transpose(2, 0, 1)
            print(data.shape)
            data = normalize(data)
            
            patches = split_patch(data, patch_size)
            self.data.extend(patches)
            
        print(f"Number data: {len(self.data)}")

    def __getitem__(self, idx):
        patch = self.data[idx]
        patch = np.expand_dims(patch, 0)
        if self.transform:
            patch = self.transform(torch.Tensor(patch))
        cond = {}
        return np.float32(patch), cond

    def __len__(self):
        return len(self.data)