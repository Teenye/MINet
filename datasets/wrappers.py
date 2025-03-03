import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import torch.nn.functional as F
from datasets import register
from utils import to_pixel_samples
from utils import make_coord

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, mean_path, std_path, lat_path, lon_path, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.lat = np.load(lat_path)
        self.lon  = np.load(lon_path)
        mean = []
        std = []
        for name in self.mean.files:
            mean.append(self.mean[name])
            std.append(self.std[name])
        self.mean = torch.from_numpy(np.stack(mean,axis=1))
        self.std = torch.from_numpy(np.stack(std,axis=1))
        self.lat = torch.from_numpy(self.lat)
        self.lon = torch.from_numpy(self.lon)
        
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt_raw': crop_hr,
            'mean': self.mean,
            'std': self.std,
            'lat': self.lat,
            'lon': self.lon,
        }


@register('sr-implicit-paired-ab')
class SRImplicitPairedAB(Dataset):

    def __init__(self, dataset, mean_path, std_path, lat_path, lon_path, ab_scale, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.ab_scale = ab_scale
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.lat = np.load(lat_path)
        self.lon  = np.load(lon_path)
        mean = []
        std = []
        for name in self.mean.files:
            mean.append(self.mean[name])
            std.append(self.std[name])
        self.mean = torch.from_numpy(np.stack(mean,axis=1))
        self.std = torch.from_numpy(np.stack(std,axis=1))
        self.lat = torch.from_numpy(self.lat)
        self.lon = torch.from_numpy(self.lon)
        
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        
        
        # ab s
        s = self.ab_scale
        # print('before,',img_lr.shape)
        img_lr = F.interpolate(img_lr.unsqueeze(0), size=(img_hr.shape[-2]//s,img_hr.shape[-1]//s),mode='bilinear',align_corners=False).squeeze(0)
        
        
        # print(img_lr.shape)
        
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt_raw': crop_hr,
            'mean': self.mean,
            'std': self.std,
            'lat': self.lat,
            'lon': self.lon,
        }


@register('sr-implicit-paired-same')
class SRImplicitPairedSame(Dataset):

    def __init__(self, dataset, mean_path, std_path, lat_path, lon_path, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.lat = np.load(lat_path)
        self.lon  = np.load(lon_path)
        mean = []
        std = []
        for name in self.mean.files:
            mean.append(self.mean[name])
            std.append(self.std[name])
        self.mean = torch.from_numpy(np.stack(mean,axis=1))
        self.std = torch.from_numpy(np.stack(std,axis=1))
        self.lat = torch.from_numpy(self.lat)
        self.lon = torch.from_numpy(self.lon)
        
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        img_lr = F.interpolate(img_lr.unsqueeze(0), size=(img_hr.shape[-2],img_hr.shape[-1]),mode='bilinear',align_corners=False).squeeze(0)
        
        
        crop_lr, crop_hr = img_lr, img_hr
        
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'gt': crop_hr,
            'mean': self.mean,
            'std': self.std,
            'lat': self.lat,
            'lon': self.lon,
        }



@register('sr-implicit-paired-3channels')
class SRImplicitPaired3(Dataset):

    def __init__(self, dataset, mean_path, std_path, lat_path, lon_path,channels, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.channels = channels
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.lat = np.load(lat_path)
        self.lon  = np.load(lon_path)
        mean = []
        std = []
        for name in self.mean.files:
            mean.append(self.mean[name])
            std.append(self.std[name])
        self.mean = torch.from_numpy(np.stack(mean,axis=1))
        self.std = torch.from_numpy(np.stack(std,axis=1))
        self.lat = torch.from_numpy(self.lat)
        self.lon = torch.from_numpy(self.lon)
        
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        
        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous(),channel=self.channels)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt_raw': crop_hr,
            'mean': self.mean,
            'std': self.std,
            'lat': self.lat,
            'lon': self.lon,
        }



# @register('sr-implicit-paired-fast')
# class SRImplicitPairedFast(Dataset):

#     def __init__(self, dataset, inp_size=None, augment=False):
#         self.dataset = dataset
#         self.inp_size = inp_size
#         self.augment = augment

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img_lr, img_hr = self.dataset[idx]

#         s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
#         if self.inp_size is None:
#             h_lr, w_lr = img_lr.shape[-2:]
#             h_hr = s * h_lr
#             w_hr = s * w_lr
#             img_hr = img_hr[:, :h_lr * s, :w_lr * s]
#             crop_lr, crop_hr = img_lr, img_hr
#         else:
#             w_lr = self.inp_size
#             x0 = random.randint(0, img_lr.shape[-2] - w_lr)
#             y0 = random.randint(0, img_lr.shape[-1] - w_lr)
#             crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
#             w_hr = w_lr * s
#             x1 = x0 * s
#             y1 = y0 * s
#             crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

#         if self.augment:
#             hflip = random.random() < 0.5
#             vflip = random.random() < 0.5
#             dflip = random.random() < 0.5

#             def augment(x):
#                 if hflip:
#                     x = x.flip(-2)
#                 if vflip:
#                     x = x.flip(-1)
#                 if dflip:
#                     x = x.transpose(-2, -1)
#                 return x

#             crop_lr = augment(crop_lr)
#             crop_hr = augment(crop_hr)

#         hr_coord = make_coord([h_hr, w_hr], flatten=False)
#         hr_rgb = crop_hr
        
#         if self.inp_size is not None:
#             x0 = random.randint(0, h_hr - h_lr)
#             y0 = random.randint(0, w_hr - w_lr)
            
#             hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
#             hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
#         cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

#         return {
#             'inp': crop_lr,
#             'coord': hr_coord,
#             'cell': cell,
#             'gt': hr_rgb
#         }
    
    
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BILINEAR)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled-climax')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, mean_path, std_path, lat_path, lon_path,channels, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.channels = channels
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.lat = np.load(lat_path)
        self.lon  = np.load(lon_path)
        mean = []
        std = []
        for name in self.mean.files:
            mean.append(self.mean[name])
            std.append(self.std[name])
        self.mean = torch.from_numpy(np.stack(mean,axis=1))
        self.std = torch.from_numpy(np.stack(std,axis=1))
        self.lat = torch.from_numpy(self.lat)
        self.lon = torch.from_numpy(self.lon)
        self.scale_min = 1
        self.scale_max = 4
        
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = F.interpolate(img.unsqueeze(0), size=(h_lr,w_lr), mode='bilinear', align_corners=False).squeeze(0)
 
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = F.interpolate(img.unsqueeze(0), size=(w_lr,w_lr), mode='bilinear', align_corners=False).squeeze(0)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous(),channel=self.channels)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'mean': self.mean,
            'std': self.std,
            'lat': self.lat,
            'lon': self.lon,
        }



@register('sr-implicit-downsampled-single')
class SRImplicitDownsampledSingle(Dataset):

    def __init__(self, dataset, mean_path, std_path, lat_path, lon_path,channels, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.lat = np.load(lat_path)
        self.lon  = np.load(lon_path)
        mean = []
        std = []
        for name in self.mean.files:
            mean.append(self.mean[name])
            std.append(self.std[name])
        self.mean = torch.from_numpy(np.stack(mean,axis=1))
        self.std = torch.from_numpy(np.stack(std,axis=1))
        self.lat = torch.from_numpy(self.lat)
        self.lon = torch.from_numpy(self.lon)
        self.channels = channels
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_hr = self.dataset[idx]
        base_s = 2
        if base_s != 1:
            h_lr = math.floor(img_hr.shape[-2] / base_s + 1e-9)
            w_lr = math.floor(img_hr.shape[-1] / base_s + 1e-9)
            img_hr = img_hr[:, :round(h_lr * base_s), :round(w_lr * base_s)] # assume round int
            img_hr = F.interpolate(img_hr.unsqueeze(0), size=(h_lr,w_lr), mode='bilinear', align_corners=False).squeeze(0)
            self.lat = F.interpolate(self.lat.unsqueeze(0).unsqueeze(0), size=h_lr).squeeze(0).squeeze(0)
            self.lon = F.interpolate(self.lon.unsqueeze(0).unsqueeze(0), size=w_lr).squeeze(0).squeeze(0)
        
        # scale尺度
        s = 2
        if self.inp_size is None:
            h_lr = math.floor(img_hr.shape[-2] / s + 1e-9)
            w_lr = math.floor(img_hr.shape[-1] / s + 1e-9)
            img_hr = img_hr[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_lr = F.interpolate(img_hr.unsqueeze(0), size=(h_lr,w_lr), mode='bilinear', align_corners=False).squeeze(0)

            crop_lr, crop_hr = img_lr, img_hr

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous(),channel=self.channels)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt_raw': crop_hr,
            'mean': self.mean,
            'std': self.std,
            'lat': self.lat,
            'lon': self.lon,
        }



# @register('sr-implicit-downsampled')
# class SRImplicitDownsampled(Dataset):

#     def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
#                  augment=False, sample_q=None, mean_path=None, std_path=None, lat_path=None):
#         self.dataset = dataset
#         self.inp_size = inp_size
#         self.scale_min = scale_min
#         if scale_max is None:
#             scale_max = scale_min
#         self.scale_max = scale_max
#         self.augment = augment
#         self.sample_q = sample_q
        
        
#         self.mean = np.load(mean_path)
#         self.std = np.load(std_path)
#         self.lat = np.load(lat_path)
#         mean = []
#         std = []
#         for name in self.mean.files:
#             mean.append(self.mean[name])
#             std.append(self.std[name])
#         self.mean = torch.from_numpy(np.stack(mean,axis=1))
#         self.std = torch.from_numpy(np.stack(std,axis=1))
#         self.lat = torch.from_numpy(self.lat)
#         print(self.mean.shape, self.std.shape,self.lat.shape)
        
        
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img = self.dataset[idx]
#         s = random.uniform(self.scale_min, self.scale_max)

#         if self.inp_size is None:
#             h_lr = math.floor(img.shape[-2] / s + 1e-9)
#             w_lr = math.floor(img.shape[-1] / s + 1e-9)
#             img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
#             img_down = F.interpolate(img, size=(h_lr,w_lr), mode='bilinear', align_corners=False)

#             crop_lr, crop_hr = img_down, img
#         else:
#             w_lr = self.inp_size
#             w_hr = round(w_lr * s)
#             x0 = random.randint(0, img.shape[-2] - w_hr)
#             y0 = random.randint(0, img.shape[-1] - w_hr)
#             crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
#             crop_lr = resize_fn(crop_hr, w_lr)

#         if self.augment:
#             hflip = random.random() < 0.5
#             vflip = random.random() < 0.5
#             dflip = random.random() < 0.5

#             def augment(x):
#                 if hflip:
#                     x = x.flip(-2)
#                 if vflip:
#                     x = x.flip(-1)
#                 if dflip:
#                     x = x.transpose(-2, -1)
#                 return x

#             crop_lr = augment(crop_lr)
#             crop_hr = augment(crop_hr)

#         hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous(),channel=1)

#         if self.sample_q is not None:
#             sample_lst = np.random.choice(
#                 len(hr_coord), self.sample_q, replace=False)
#             hr_coord = hr_coord[sample_lst]
#             hr_rgb = hr_rgb[sample_lst]

#         cell = torch.ones_like(hr_coord)
#         cell[:, 0] *= 2 / crop_hr.shape[-2]
#         cell[:, 1] *= 2 / crop_hr.shape[-1]

#         return {
#             'inp': crop_lr,
#             'coord': hr_coord,
#             'cell': cell,
#             'gt': hr_rgb,
#             'mean': self.mean,
#             'std': self.std,
#             'lat': self.lat,
#         }


@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }    
    
    
@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
