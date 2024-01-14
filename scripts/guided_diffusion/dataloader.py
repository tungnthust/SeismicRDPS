import torch
import numpy as np

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    image = image * 2 - 1
    return image

class F3Dataset(torch.utils.data.Dataset):
    def __init__(self, directory, mode="train", patch_size=128, stride=48, transform=None):
        
        super().__init__()
        self.data = None
        if mode == "train":
            data1 = np.load(f'{directory}/train/train_seismic.npy')
            data2 = np.load(f'{directory}/test/test1_seismic.npy')
            self.data = np.concatenate((data2, data1), axis=0).transpose(2, 0, 1)
        else:
            try:
                data = np.load(f'{directory}/test/test2_seismic.npy')
            except:
                data = np.load(f'{directory}/test_once/test2_seismic.npy')
            self.data = data.transpose(2,0,1)
        self.data = normalize(self.data)
        self.transform = transform
        self.depth = int(self.data.shape[0])
        self.num_ilines = int(self.data.shape[1])
        self.num_xlines = int(self.data.shape[2])

        self.patch_size = patch_size
        self.stride = stride
        self.num_patch_row = np.ceil((self.depth - self.patch_size) / self.stride + 1)
        self.num_patch_col_ilines = np.ceil((self.num_xlines - self.patch_size) / self.stride + 1)
        self.num_patch_col_xlines = np.ceil((self.num_ilines - self.patch_size) / self.stride + 1)
        self.num_patch_per_iline = self.num_patch_row * self.num_patch_col_ilines
        self.num_patch_per_xline = self.num_patch_row * self.num_patch_col_xlines
        self.num_patch_ilines = self.num_patch_per_iline * self.num_ilines
        self.num_patch_xlines = self.num_patch_per_xline * self.num_xlines
        self.num_patch = int(self.num_patch_ilines + self.num_patch_xlines)
        print(f"Number data: {self.num_patch}")

    def __getitem__(self, idx):
        if idx < self.num_patch_ilines:
            line_idx = idx // self.num_patch_per_iline
            patch_idx = idx % self.num_patch_per_iline
            row_idx = patch_idx // self.num_patch_col_ilines
            col_idx = patch_idx % self.num_patch_col_ilines
            if row_idx == self.num_patch_row - 1:
                end_row = self.depth
                start_row = self.depth - self.patch_size
            else:
                start_row = row_idx * 48
                end_row = start_row + self.patch_size
            if col_idx == self.num_patch_col_ilines - 1:
                end_col = self.num_xlines
                start_col = self.num_xlines - self.patch_size 
            else:
                start_col = col_idx * 48
                end_col = start_col + self.patch_size
            
            patch = self.data[int(start_row):int(end_row), int(line_idx), int(start_col):int(end_col)]
        else:
            idx = idx - self.num_patch_ilines
            line_idx = idx // self.num_patch_per_xline
            patch_idx = idx % self.num_patch_per_xline
            row_idx = patch_idx // self.num_patch_col_xlines
            col_idx = patch_idx % self.num_patch_col_xlines
            if row_idx == self.num_patch_row - 1:
                end_row = self.depth
                start_row = self.depth - self.patch_size
            else:
                start_row = row_idx * 48
                end_row = start_row + self.patch_size
            if col_idx == self.num_patch_col_xlines - 1:
                end_col = self.num_ilines
                start_col = self.num_ilines - self.patch_size 
            else:
                start_col = col_idx * 48
                end_col = start_col + self.patch_size
            patch = self.data[int(start_row):int(end_row), int(start_col):int(end_col), int(line_idx)]
        patch = np.expand_dims(patch, 0)
        if self.transform:
            patch = self.transform(torch.Tensor(patch))
        cond = {}
        return np.float32(patch), cond

    def __len__(self):
        return self.num_patch