import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from PIL import Image

def populate_train_list(limages_path, extend = 'npy'):
    image_list = glob.glob(limages_path + "/*." + extend)
    image_list = sorted(image_list)
    return image_list

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

class CapturedDataLoader(object):
    def __init__(self,image1_folder):
        self.testing_samples = CapturedDataset(image1_folder)
        self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)


class CapturedDataset(Dataset): 
    def __init__(self, image1_folder):
        self.image1_list = populate_train_list(image1_folder, extend = 'npy')
        self.len = len(self.image1_list)
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __getitem__(self,idx):
        file_name = self.image1_list[idx]
        file = np.load(file_name,allow_pickle=True)
        depth = self.totensor(file.take(0)['depth'])
        image = self.totensor(file.take(0)['image'])
        image = self.normalize(image)
        spad = file.take(0)['transient']
        # padding for max depth
        if len(spad)<3000:
            spad = np.concatenate((np.zeros(64)+spad.min(),spad[:-188]))
        else:
            spad = np.concatenate((np.zeros(126)+spad.min(),spad,np.zeros(860)+spad.min()))
        spad = downsample(spad,1024)
        spad = (spad-spad.min())/(spad.max()-spad.min())
        spad = torch.Tensor(spad)
        
        return {'image': image, 'depth': depth, 'spad': spad, 'file_name':file_name, 'image_path':file_name}
    
    def __len__(self):
        return self.len
