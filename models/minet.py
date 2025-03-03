import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import math
from models import register
from utils import make_coord
import numpy as np
from fmoe import FMoETransformerMLP
from fmoe.layers import FMoE
from fmoe.gates import NaiveGate,SwitchGate,MyGate

class Expert5c(nn.Module):
    def __init__(self, d_model, hidden_size=256, output_size=5):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(d_model, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, output_size)
                                    )

    def forward(self, x, *args):  # *args is necessary
        x = self.layers(x)
        return x

class Expert3c(nn.Module):
    def __init__(self, d_model, hidden_size=256, output_size=3):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(d_model, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, output_size)
                                    )

    def forward(self, x, *args):  # *args is necessary
        x = self.layers(x)
        return x

class Expert1c(nn.Module):
    def __init__(self, d_model, hidden_size=256, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(d_model, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, hidden_size)
                                    ,nn.ReLU()
                                    ,nn.Linear(hidden_size, output_size)
                                    )

    def forward(self, x, *args):  # *args is necessary
        x = self.layers(x)
        return x

@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
            
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                spaces = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(spaces + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, spaces in zip(preds, areas):
            ret = ret + pred * (spaces / tot_area).unsqueeze(-1)
            
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

# 现在的方法
@register('minet')
class MINet(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.encoder = models.make(encoder_spec)
        
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
                # LR->HR  对应1-》3快
            self.upsample_1_2 = nn.Sequential(
                nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.upsample_2_4 = nn.Sequential(
                nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.upsample_4_8 = nn.Sequential(
                nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
                nn.PixelShuffle(2)
            )
           
            imnet_in_dim = 4 * imnet_in_dim
            imnet_in_dim += 2 # attach coord
            
            
            # ------------------FN----------------------------
            # self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
            
            # ------------------MOE-----------------------------
            channels = imnet_spec['args']['out_dim']
            if channels == 5:
                self.imnet = FMoE(num_expert=8, d_model=imnet_in_dim, expert=Expert5c, gate=NaiveGate, top_k=2)
            # elif channels == 3:
            #     self.imnet = FMoE(num_expert=8, d_model=imnet_in_dim, expert=Expert3c, gate=NaiveGate, top_k=2)
            # elif channels == 1:
            #     self.imnet = FMoE(num_expert=8, d_model=imnet_in_dim, expert=Expert1c, gate=NaiveGate, top_k=2)
                
        else:
            self.imnet = None
           
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat_1x = self.feat
        # FPN
        feat_2x = self.upsample_1_2(feat_1x) # 1->2
        feat_4x = self.upsample_2_4(feat_2x) # 2->3
        feat_8x = self.upsample_4_8(feat_4x) # 3->4
        
        if self.imnet is None:
            ret = F.grid_sample(feat_1x, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret
            
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat_1x.shape[-2] / 2
        ry = 2 / feat_1x.shape[-1] / 2

        feat_1x_coord = make_coord(feat_1x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat_1x.shape[0], 2, *feat_1x.shape[-2:])
            
            
        preds = []
        areas = []
        bs, q = coord.shape[:2]
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat1x = F.grid_sample(
                    feat_1x, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat2x = F.grid_sample(
                    feat_2x, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat4x = F.grid_sample(
                    feat_4x, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat8x = F.grid_sample(
                    feat_8x, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_1x_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                
                # 多尺度特征
                inp = torch.cat([q_feat1x,q_feat2x,q_feat4x,q_feat8x], dim=-1)
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat_1x.shape[-2]
                rel_coord[:, :, 1] *= feat_1x.shape[-1]
                
                inp = torch.cat([inp, rel_coord],dim=-1)
                

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                spaces = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(spaces + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, spaces in zip(preds, areas):
            ret = ret + pred * (spaces / tot_area).unsqueeze(-1)
            
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)    
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
   
