import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register

from pathlib import Path


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file)))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

@register('numpy-folder')
class NumpyFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    np.load(file)).float())

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
    
@register('paired-numpy-folders')
class PairedNumpyFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = NumpyFolder(root_path_1, **kwargs)
        self.dataset_2 = NumpyFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]



# climax预处理的npydata为数据
@register('weather-numpy-folder')
class WNumpyFolder(Dataset):

    def __init__(self, root_path, per_num_samples, split_file=None, split="in",
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.per_num_samples = per_num_samples
        if split_file is None:
            filenames = sorted(os.listdir(root_path))
            
        self.files = []
        self.num_samples = 0
        pre_root_path = Path(root_path).parent
        self.mean = os.path.join(pre_root_path,'normalize_mean_'+split+'put.npz')
        self.std = os.path.join(pre_root_path,'normalize_std_'+split+'put.npz')
        for filename in filenames:
            if split == 'in':
                if filename.endswith('inp.npz'):
                    file = os.path.join(root_path, filename)
                    self.files.append(file)
                    self.num_samples += self.per_num_samples
                        
            elif split == 'out':
                if filename.endswith('out.npz'):
                    file = os.path.join(root_path, filename)
                    self.files.append(file)
                    self.num_samples += self.per_num_samples
                    
    def __len__(self):
        return self.num_samples * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.num_samples
        file_idx = idx // self.per_num_samples
        npy_idx = idx % self.per_num_samples
        x = np.load(self.files[file_idx])
        mean_all = np.load(self.mean)
        std_all =  np.load(self.std)
        arrays = []
        # 多种变量 -》 串接为 3通道
        for name in mean_all.keys():
            mean = mean_all[name]
            std = std_all[name]
            arrays.append((x[name]-mean)/std)
        # 沿第 1 维度串接所有数组
        x_c3 = np.concatenate(arrays, axis=1)
        
        x_tensor = torch.from_numpy(x_c3)[npy_idx]
        return x_tensor

@register('paired-weather-numpy-folders')
class WPairedNumpyFolders(Dataset):

    def __init__(self, root_path_1, root_path_2,split_1,split_2, per_num_samples, **kwargs):
        self.dataset_1 = WNumpyFolder(root_path_1,split=split_1,per_num_samples=per_num_samples, **kwargs)
        self.dataset_2 = WNumpyFolder(root_path_2,split=split_2,per_num_samples=per_num_samples, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

    
# @register('weather-single-numpy-folder')
# class WSNumpyFolder(Dataset):
#     def __init__(self, root_path, split_file=None,
#                  repeat=1, cache='none'):
#         self.repeat = repeat
#         self.cache = cache

#         if split_file is None:
#             filenames = sorted(os.listdir(root_path))
            
#         self.files = []
#         self.num_samples = 0
#         pre_root_path = Path(root_path).parent
#         self.mean = os.path.join(pre_root_path,'normalize_mean.npz')
#         self.std = os.path.join(pre_root_path,'normalize_std.npz')
#         for filename in filenames:
#             file = os.path.join(root_path, filename)
#             self.files.append(file)
#             self.num_samples += 546
        
#     def __len__(self):
#         return self.num_samples * self.repeat

#     def __getitem__(self, idx):
#         idx = idx % self.num_samples
#         file_idx = idx // 546
#         npy_idx = idx % 546
#         x = np.load(self.files[file_idx])
#         mean_all = np.load(self.mean)
#         std_all =  np.load(self.std)
#         arrays = []
#         # 多种变量 -》 串接为 3通道
#         for name in ['2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind']:
#             mean = mean_all[name]
#             std = std_all[name]
#             arrays.append((x[name]-mean)/std)
#         # 沿第 1 维度串接所有数组
#         x_c3 = np.concatenate(arrays, axis=1)
        
#         x_tensor = torch.from_numpy(x_c3)[npy_idx]
#         return x_tensor

@register('weather-single-numpy-folder')
class WSNumpyFolder(Dataset):
    def __init__(self, root_path, per_nums_samples, split_file=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
            
        self.files = []
        self.num_samples = 0
        pre_root_path = Path(root_path).parent
        self.mean = os.path.join(pre_root_path,'normalize_mean.npz')
        self.std = os.path.join(pre_root_path,'normalize_std.npz')
        self.per_nums_samples = per_nums_samples
        for filename in filenames:
            file = os.path.join(root_path, filename)
            self.files.append(file)
            self.num_samples += self.per_nums_samples
        
    def __len__(self):
        return self.num_samples * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.num_samples
        file_idx = idx // self.per_nums_samples
        npy_idx = idx % self.per_nums_samples
        x = np.load(self.files[file_idx])
        mean_all = np.load(self.mean)
        std_all =  np.load(self.std)
        arrays = []
        for name in ['2m_temperature','temperature_850','geopotential_500']:
            mean = mean_all[name]
            std = std_all[name]
            arrays.append((x[name]-mean)/std)
        x_c3 = np.concatenate(arrays, axis=1)
        x_tensor = torch.from_numpy(x_c3)[npy_idx]
        return x_tensor

# 3通道
# climax预处理的npydata为数据
@register('weather-numpy-folder3')
class WNumpyFolder3(Dataset):

    def __init__(self, root_path, per_num_samples, split_file=None, split="in",
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.per_num_samples = per_num_samples
        if split_file is None:
            filenames = sorted(os.listdir(root_path))
            
        self.files = []
        self.num_samples = 0
        pre_root_path = Path(root_path).parent
        self.mean = os.path.join(pre_root_path,'normalize_mean_'+split+'put.npz')
        self.std = os.path.join(pre_root_path,'normalize_std_'+split+'put.npz')
        for filename in filenames:
            if split == 'in':
                if filename.endswith('inp.npz'):
                    file = os.path.join(root_path, filename)
                    self.files.append(file)
                    self.num_samples += self.per_num_samples
                        
            elif split == 'out':
                if filename.endswith('out.npz'):
                    file = os.path.join(root_path, filename)
                    self.files.append(file)
                    self.num_samples += self.per_num_samples
                    
    def __len__(self):
        return self.num_samples * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.num_samples
        file_idx = idx // self.per_num_samples
        npy_idx = idx % self.per_num_samples
        x = np.load(self.files[file_idx])
        mean_all = np.load(self.mean)
        std_all =  np.load(self.std)
        arrays = []
        # 多种变量 -》 串接为 3通道
        for name in ['2m_temperature','geopotential_500', 'temperature_850']:
            mean = mean_all[name]
            std = std_all[name]
            arrays.append((x[name]-mean)/std)
        # 沿第 1 维度串接所有数组
        x_c3 = np.concatenate(arrays, axis=1)
        
        x_tensor = torch.from_numpy(x_c3)[npy_idx]
        return x_tensor



@register('paired-weather-numpy-folders-3channels')
class WPairedNumpyFolders3(Dataset):

    def __init__(self, root_path_1, root_path_2,split_1,split_2, per_num_samples, **kwargs):
        self.dataset_1 = WNumpyFolder3(root_path_1,split=split_1,per_num_samples=per_num_samples, **kwargs)
        self.dataset_2 = WNumpyFolder3(root_path_2,split=split_2,per_num_samples=per_num_samples, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
