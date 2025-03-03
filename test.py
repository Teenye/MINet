import argparse
import os
import math
from functools import partial
from scipy import stats
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom
import torch.nn.functional as F
import pandas as pd
import xarray as xr
import datasets
import models
import utils
import numpy as np
from skimage.transform import resize

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def lat_weighted_rmse(pred, y, lat):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    error = (pred - y) ** 2  # [B, V, H, W]
    # lattitude weights
    w_lat = torch.cos(lat * (torch.pi / 180.0))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = w_lat.unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    rmse_list = []
    with torch.no_grad():
        for i in range(error.shape[1]):
            rmse_list.append(torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=-2))
            ))
            
    return rmse_list

def rmse(pred, y):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    error = (pred - y) ** 2  # [B, V, H, W]
    # lattitude weights
  
    rmse_list = []
    with torch.no_grad():
        for i in range(error.shape[1]):
            rmse_list.append(torch.sqrt(torch.mean(error[:, i])))
            
    return rmse_list

def pearson(pred, y):
    """
    y: [B, V, H, W]
    pred: [B, V, H, W]
    vars: list of variable names
    lat: H
    """
    p_list = []
    with torch.no_grad():
        for i in range(pred.shape[1]):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            p_list.append(stats.pearsonr(pred_.cpu().numpy(), y_.cpu().numpy())[0])
    return p_list


def mean_bias(pred, y):
    """
    y: [B, V, H, W]
    pred: [B, V, H, W]
    vars: list of variable names
    lat: H
    """
    m_b_climax = []
    with torch.no_grad():
        for i in range(pred.shape[1]):
            pred_, y_ = pred[:, i].flatten(), y[:, i].flatten()
            m_b_climax.append(pred_.mean() - y_.mean())
    return m_b_climax


def cal_3metric(x, y, lat, mean, std):
    '''RMSE Pearson Mean-bias'''
    
    B,V,H,W = y.shape
    x = x.view(B,H,W,V).permute(0,3,1,2)
    
    # denormalize
    x = x * std.squeeze(1).unsqueeze(-1).unsqueeze(-1) + mean.squeeze(1).unsqueeze(-1).unsqueeze(-1)
    y = y * std.squeeze(1).unsqueeze(-1).unsqueeze(-1) + mean.squeeze(1).unsqueeze(-1).unsqueeze(-1)
    
    rmse_list = lat_weighted_rmse(x,y,lat)
    # rmse_list = rmse(x,y)
    
    p_list = pearson(x,y)
    m_b_list = mean_bias(x,y)
    
    return  rmse_list, p_list, m_b_list
    


def eval_metric(var_names, loader, model, data_norm=None, scale_max=4, save_nc=False):
    model.eval()
    var_num = len(var_names)
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    scale = 4 #当前放大倍数
    
    rmse_list = []
    p_list = []
    m_b_list = []
    res_list = []
    
    for i in range(var_num):
        rmse_list.append(utils.Averager())
        p_list.append(utils.Averager())
        m_b_list.append(utils.Averager())
    
    pbar = tqdm(loader, leave=False, desc='val')
    
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()


        inp = (batch['inp'] - inp_sub) / inp_div
        
        
        mean =batch['mean']
        std = batch['std']
        lat = batch['lat']
        lon = batch['lon']
        
        gt = (batch['gt_raw'] - inp_sub) / inp_div
        gt_shape = gt.shape[-2:]
        coord = batch['coord']
        cell = batch['cell']
                       
        with torch.no_grad():
            pred = model(inp, coord, cell*max(scale/scale_max, 1)) 
        
        
        # 双线性插值
        # pred = F.interpolate(inp, size=(gt_shape),mode='bicubic',align_corners=False)
        # pred = pred.permute(0, 2, 3, 1)
        
        # HR
        # pred = gt
        # pred = pred.permute(0, 2, 3, 1)
        
        
        pred = pred* gt_div + gt_sub
        
        rmse, p, m_b = cal_3metric(pred, gt,lat, mean, std)
        
        B, V, H, W = gt.shape
        if save_nc: #是否保存为nc文件
            res_list.append(pred.squeeze(0))
            
        
    # -------------------------   
        for index in range(var_num):
            rmse_list[index].add(rmse[index].item(), inp.shape[0])
            p_list[index].add(p[index].item(), inp.shape[0])
            m_b_list[index].add(m_b[index].item(), inp.shape[0])

    # 对batch平均
    res_rmse_list = []
    res_p_list = []
    res_m_b_list =  []
    for index in range(var_num):
        res_rmse_list.append(rmse_list[index].item())
        res_p_list.append(p_list[index].item())
        res_m_b_list.append(m_b_list[index].item())
    
    if save_nc:
        res_data = torch.stack(res_list, dim=0)
        res_data = res_data.reshape(2880, H, W, V).cpu().numpy()
        
        output_time1 = pd.date_range(start='2013-01-01 06:00', periods=1440, freq='6H')
        output_time2 = pd.date_range(start='2014-01-01 06:00', periods=1440, freq='6H')
        output_time = output_time1.append(output_time2)

        output_lat = lat.squeeze(0).cpu().numpy()
        output_lon = lon.squeeze(0).cpu().numpy()
        
        
        output_mean = mean.squeeze(0).squeeze(0).cpu().numpy()
        output_std = std.squeeze(0).squeeze(0).cpu().numpy()
        
        # 使用循环会出错        
        res = res_data[:,:,:,0]*output_std[0] + output_mean[0]
        da = xr.DataArray(res, coords=[output_time,output_lat, output_lon], dims=['time', 'lat', 'lon'])
        da.to_netcdf(save_dir+'/t2m_res.nc')
        
        res = res_data[:,:,:,1]*output_std[1] + output_mean[1]
        db = xr.DataArray(res, coords=[output_time,output_lat, output_lon], dims=['time', 'lat', 'lon'])
        db.to_netcdf(save_dir+'/u10_res.nc')
        
        res = res_data[:,:,:,2]*output_std[2] + output_mean[2]
        dc = xr.DataArray(res, coords=[output_time,output_lat, output_lon], dims=['time', 'lat', 'lon'])
        dc.to_netcdf(save_dir+'/v10_res.nc')
        
        res = res_data[:,:,:,3]*output_std[3] + output_mean[3]
        da = xr.DataArray(res, coords=[output_time,output_lat, output_lon], dims=['time', 'lat', 'lon'])
        da.to_netcdf(save_dir+'/z500_res.nc')
        
        res = res_data[:,:,:,4]*output_std[4] + output_mean[4]
        db = xr.DataArray(res, coords=[output_time,output_lat, output_lon], dims=['time', 'lat', 'lon'])
        db.to_netcdf(save_dir+'/t850_res.nc')
    

    return  res_rmse_list, res_p_list, res_m_b_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--save_res', action='store_true', help="If specified, save the results.")
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    
    if args.save_res:
        save_dir = './save_res/'+model_spec['name']
        os.makedirs(save_dir)
    
    model = models.make(model_spec, load_sd=True).cuda()

    var_names = config['var_names']
    rmse_list, p_list, m_b_list = eval_metric(var_names, loader, model,
        data_norm=config.get('data_norm'),
        scale_max = int(args.scale_max), save_nc = args.save_res)
    
    for index, var_name in enumerate(var_names):
        print('val: {} rmse: {:.4f}, p: {:.4f}, m_b: {:.4f}\n'.format(var_name,rmse_list[index],p_list[index],m_b_list[index]))
                    