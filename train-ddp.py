# modified from: https://github.com/yinboc/liif

import argparse
import os
import warnings
# 屏蔽特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)

import gc
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
# from test_era5 import eval_metric
from test import eval_metric

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

def init_distributed_mode(backend='nccl'):
    dist.init_process_group(backend=backend)
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    return local_rank

def make_data_loader(local_rank, spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))
    
    if tag == 'train':
        sampler = DistributedSampler(dataset)
    else: 
        sampler = None
    
    loader = DataLoader(dataset, batch_size=spec['batch_size'],sampler=sampler, num_workers=4, pin_memory=True)
    return loader, sampler


def make_data_loaders(local_rank):
    train_loader, train_sampler = make_data_loader(local_rank, config.get('train_dataset'), tag='train')
    val_loader, _  = make_data_loader(local_rank, config.get('val_dataset'), tag='val')
    return train_loader, val_loader, train_sampler


def prepare_training(local_rank):
#     if config.get('resume') is not None:
    if os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            
    # 初始化GradScaler
    scaler = GradScaler()

    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler, scaler

def lat_w_loss(x,y,lat):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    error = (x - y) ** 2  # [B, V, H, W]
    # lattitude weights
    w_lat = torch.cos(lat * (torch.pi / 180.0))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = w_lat.unsqueeze(1).unsqueeze(-1).to(dtype=error.dtype, device=error.device) # (1,h,1)
    
    loss = (error * w_lat).mean()

    return loss

def train(train_loader,train_sampler, model, optimizer,scaler, epoch, local_rank):
    model.train()
    train_loss = utils.Averager()
    loss_fn = nn.L1Loss()
    data_norm = config['data_norm']
    
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    
    num_dataset = 46080
    iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
                        * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    
    t = tqdm(train_loader, leave=False, desc='train')
    for batch in t:
        train_sampler.set_epoch(epoch)  # Ensure randomness
        for k, v in batch.items():
            batch[k] = v.to(local_rank)
            
        inp = (batch['inp'] - inp_sub) / inp_div
        
        
        gt = (batch['gt_raw'] - gt_sub) / gt_div
        
        # 
        lat = batch['lat']
        
        optimizer.zero_grad()
        
        
        pred = model(inp, batch['coord'], batch['cell'])
        
        B,V,H,W = gt.shape
        pred = pred.view(B,H,W,V).permute(0,3,1,2)
        
        # loss = lat_w_loss(pred, gt,lat) # lat-weighted loss
        loss = loss_fn(pred,gt) # l1-loss
        # with autocast():
        # loss = loss_fn(pred,gt)
        
        # tensorboard
        if local_rank == 0:
            writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        
        train_loss.add(loss.item())
        t.set_postfix(loss=loss.item())
            
        # writer.add_scalars('psnr', {'train': psnr}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        
        loss.backward()
        optimizer.step()
        
        #  # 反向传播和优化
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        pred = None; loss = None
        
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    # Initialize distributed mode
    local_rank = init_distributed_mode()

    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        

    train_loader, val_loader, train_sampler = make_data_loaders(local_rank)
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler, scaler = prepare_training(local_rank)

    
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        # model = nn.parallel.DataParallel(model)
        model = DDP(model, device_ids=[local_rank])

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = 1e18 #最好结果
    var_names = config['var_names']
    channels = len(var_names)
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        
        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        lr = optimizer.param_groups[0]['lr']
        # train_loss = 0
        train_loss = train(train_loader, train_sampler, model, optimizer,scaler, epoch, local_rank) 
        if lr_scheduler is not None:
            lr_scheduler.step()

        if local_rank == 0:
            log_info.append('train: loss={:.4f},lr={:.6f}'.format(train_loss,lr))
#         writer.add_scalars('loss', {'train': train_loss}, epoch)

        if local_rank == 0:
            model_ = model.module
            model_spec = config['model']
            model_spec['sd'] = model_.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'epoch': epoch
            }

            torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(sv_file,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

            if (epoch_val is not None) and (epoch % epoch_val == 0):
                rmse_list, p_list, m_b_list  = eval_metric(var_names, val_loader, model,
                    data_norm=config['data_norm'])
    
                # 不同变量
                for index, var_name in enumerate(var_names):
                    log_info.append('val: {} rmse: {:.4f}, p: {:.4f}, m_b: {:.4f}\n'.format(var_name,rmse_list[index],p_list[index],m_b_list[index]))
                
                rmse_sum = 0
                for index, var_name in enumerate(var_names):
                    rmse_sum += rmse_list[index]
                rmse_sum = rmse_list[0] + rmse_list[1] + rmse_list[2] + rmse_list[4]
                
                if rmse_sum  < max_val_v:
                    max_val_v = rmse_sum
                    torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        
        if local_rank == 0:
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
            log(', '.join(log_info))
            writer.flush()
     # Cleanup
    
    dist.destroy_process_group()
    del model
    # 调用垃圾回收
    gc.collect()

    # # 清空缓存
    # torch.cuda.empty_cache()
    # spec = config['test_dataset']
    # dataset = datasets.make(spec['dataset'])
    # dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    # loader = DataLoader(dataset, batch_size=spec['batch_size'],
    #     num_workers=8, pin_memory=True)
    
    
    # model_spec = torch.load(save_path+'/epoch-best.pth')['model']
    # model = models.make(model_spec, load_sd=True).cuda()
    
    # var_names = config['var_names']
    # rmse_list, p_list, m_b_list = eval_metric(var_names, loader, model,
    #     data_norm=config.get('data_norm'),
    #     scale_max = 4, save_nc = False)
    
    # if local_rank == 0:
    #     for index, var_name in enumerate(var_names):
    #         print('test: {} rmse: {:.4f}, p: {:.4f}, m_b: {:.4f}\n'.format(var_name,rmse_list[index],p_list[index],m_b_list[index]))
                        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument('--local_rank', type=int, help='local rank passed from distributed launcher')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    torch.backends.cudnn.benchmark = True  # 启用自动优化，提升性能
    torch.backends.cudnn.deterministic = False  # 允许使用非确定性算法

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    
    main(config, save_path)
    
    