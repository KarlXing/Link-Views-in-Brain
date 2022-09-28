import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
import json

class WebotDataset(Dataset):
    def __init__(self, sim_path, camera_path, label_path, mask_path=None, device = torch.device('cpu')):
        self.sim_data = torch.from_numpy(np.load(sim_path)).float().to(device).permute(0,3,1,2)  # B x 3 x 128 x 140
        self.camera_data = torch.from_numpy(np.load(camera_path)).float().to(device).permute(0,3,1,2) # B x 3 x 64 x 64
        self.label = torch.from_numpy(np.load(label_path)).float().to(device)
        self.mask = torch.from_numpy(np.load(mask_path)).float().to(device) if mask_path else None
        self.device = device
        assert(self.sim_data.shape[0] == self.camera_data.shape[0] and self.sim_data.shape[0] == self.label.shape[0])
    
    def __len__(self):
        return self.sim_data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask = self.mask[idx] if self.mask is not None else None
        return self.sim_data[idx], self.camera_data[idx], self.label[idx], mask
    
    def get_label_dim(self):
        return self.label.shape[-1]

    def get_cam_shape(self):
        return self.camera_data.shape[1:]

    
class WebotSeqDataset(Dataset):
    def __init__(self, input_path, target_path, label_path, mask_path=None, device = torch.device('cpu')):
        self.input_data = torch.from_numpy(np.load(input_path)).float().to(device).permute(0,3,1,2)  # B x 15 x H x W
        self.target_data = torch.from_numpy(np.load(target_path)).float().to(device).permute(0,3,1,2)  # B x 3 x H x W

        self.label = torch.from_numpy(np.load(label_path)).float().to(device)
        self.mask = torch.from_numpy(np.load(mask_path)).float().to(device) if mask_path else None
        self.device = device
        assert(self.input_data.shape[0] == self.target_data.shape[0])
    
    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mask = self.mask[idx] if self.mask is not None else None
        return self.input_data[idx], self.target_data[idx], self.label[idx], mask
    
    def get_label_dim(self):
        return self.label.shape[-1]



def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class LinearScheduler:
    def __init__(self, start_time=0, start_value=1, end_time=1, end_value=1):
        self.t = 0
        self.start_time, self.start_value = start_time, start_value
        self.end_time, self.end_value = end_time, end_value
    
    def step(self):
        self.t += 1
    
    def val(self):
        return self.start_value + \
               (self.t < self.end_time and self.t >= self.start_time) * (self.t-self.start_time)/(self.end_time - self.start_time) * (self.end_value - self.start_value) + \
               (self.t >= self.end_time) * (self.end_value - self.start_value)


def save_args(args, save_path, name='config.json'):
    '''input: Argument Parser object and save path'''

    save_object = vars(args)
    save_path = os.path.join(save_path, name) if os.path.isdir(save_path) else save_path
    with open(save_path, 'w+') as f:
        json.dump(save_object, f, indent=4)