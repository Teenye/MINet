import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import math
from models import register
from utils import make_coord
import numpy as np

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

def positional_encoding(coordinates, d_model):
    """
    生成位置编码
    :param coordinates: 坐标张量，形状为 (batchsize, 4, 2)
    :param d_model: 特征维度 (64)
    :return: 位置编码后的特征张量，形状为 (batchsize, 4, 64)
    """
    batchsize, num_positions, coord_dim = coordinates.shape
    
    # 生成位置编码矩阵，形状为 (batchsize, num_positions, d_model)
    pos_enc = torch.zeros(batchsize, num_positions, d_model).to(coordinates.device)
    
    for pos in range(num_positions):
        for i in range(d_model // 2):
            div_term = math.exp(-math.log(10000.0) * (2 * i) / d_model)
            pos_enc[:, pos, 2 * i] = torch.sin(coordinates[:, pos, 0] * div_term)
            pos_enc[:, pos, 2 * i + 1] = torch.cos(coordinates[:, pos, 1] * div_term)
    
    return pos_enc

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1 ,L=32):
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        # 生成M个分支，将其添加到convs中，每个分支采用不同的卷积核和不同规模的padding，保证最终得到的特征图大小一致
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # 学习通道间依赖的全连接层
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)# 将多个分支得到的特征图进行融合
        fea_s = fea_U.mean(-1).mean(-1)# 在channel这个维度进行特征抽取
        fea_z = self.fc(fea_s)# 学习通道间的依赖关系
        # 赋权操作，由于是对多维数组赋权，所以看起来比SENet麻烦一些
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)# 定义全局平均池化
        self.max_pool = nn.AdaptiveMaxPool1d(1)# 定义全局最大池化
        
        # 定义CBAM中的通道依赖关系学习层，注意这里是使用1x1的卷积实现的，而不是全连接层
        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv1d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x.permute(0,2,1)))# 实现全局平均池化
        max_out = self.fc(self.max_pool(x.permute(0,2,1)))# 实现全局最大池化
        out = avg_out + max_out# 两种信息融合
        # 最后利用sigmoid进行赋权
        return self.sigmoid(out).permute(0,2,1)


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

# 带mask的
@register('liif-modified')
class LIIF_modified(nn.Module):

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

    def gen_feat(self, inp, mask_map=None):
        self.inp = inp
        self.mask_map = mask_map
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

                # sample mask
                q_mask = F.grid_sample(
                    self.mask_map, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                    
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat*(1-q_mask), rel_coord], dim=-1)

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
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bicubic',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell, mask_map=None):
        self.gen_feat(inp, mask_map)
        return self.query_rgb(coord, cell)

@register('liif-fpn-kan')
class LIIF_fpn_kan(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
                # LR->HR  对应1-》3快
            self.upsample_1 = nn.Sequential(
                nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.upsample_2 = nn.Sequential(
                nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.upsample_3 = nn.Sequential(
                nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            # skip 块
            self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            # HR->LR 对应4-》5快
            self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            
            imnet_in_dim = imnet_in_dim*4
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = KAN(layers_hidden=[imnet_in_dim,256,3])
        else:
            self.imnet = None
            
        
        

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        # FPN
        feat_1 = self.upsample_1(feat) # 1->2
        feat_2 = self.upsample_2(feat_1) # 2->3
        feat_3 = self.upsample_3(feat_2) # 3->4
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2) # 4->3
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1) # 3->2
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
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
                q_feat2x = F.grid_sample(
                    feat_5, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat4x = F.grid_sample(
                    feat_4, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_feat8x = F.grid_sample(
                    feat_3, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                    
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat,q_feat2x,q_feat4x,q_feat8x, rel_coord], dim=-1)

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
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)    
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

@register('liif-fpn')
class LIIF_fpn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

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
            # skip 块
            self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip3 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            # HR->LR 对应4-》5快
            self.downsample_8_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            self.downsample_4_2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            self.downsample_2_1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            
            imnet_in_dim = imnet_in_dim*4
            # self.mult_attn = nn.MultiheadAttention(64, 8, batch_first=True)
            
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
        # FPN
        feat_2 = self.upsample_1_2(feat) # 1->2
        feat_4 = self.upsample_2_4(feat_2) # 2->3
        feat_8x = self.upsample_4_8(feat_4) # 3->4
        
        feat_4x = self.downsample_8_4(feat_8x) + self.skip1(feat_4) # 4->3
        feat_2x = self.downsample_4_2(feat_4x) + self.skip2(feat_2) # 3->2
        feat_1x = self.downsample_2_1(feat_2x) + self.skip3(feat)
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
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
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        preds = []
        areas = []
        bs, q = coord.shape[:2]
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
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
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                
                # 多尺度特征
                inp = torch.cat([q_feat,q_feat2x,q_feat4x,q_feat8x], dim=-1)
                
                
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_coord],dim=-1)
                
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
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)    
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

@register('liif-fpn-attn')
class LIIF_fpn_attn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

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
            # skip 块
            self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip3 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            # HR->LR 对应4-》5快
            self.downsample_8_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            self.downsample_4_2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            self.downsample_2_1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            
            imnet_in_dim = imnet_in_dim*4
            
            self.sp_attn = nn.MultiheadAttention(64, 8, batch_first=True)
        
            self.sc_attn = nn.MultiheadAttention(64, 8, batch_first=True)
            
            # # 可学习的query嵌入
            # self.query_embed = nn.Parameter(torch.randn(1, 64))  # 可学习的 Query 嵌入
            # 将坐标映射到 d_model 维度
            # self.coord_projection = nn.Linear(2, 64)

            
            
            # imnet_in_dim += 2 # attach coord
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
        # FPN
        feat_2 = self.upsample_1_2(feat) # 1->2
        feat_4 = self.upsample_2_4(feat_2) # 2->3
        feat_8x = self.upsample_4_8(feat_4) # 3->4
        
        feat_4x = self.downsample_8_4(feat_8x) + self.skip1(feat_4) # 4->3
        feat_2x = self.downsample_4_2(feat_4x) + self.skip2(feat_2) # 3->2
        feat_1x = self.downsample_2_1(feat_2x) + self.skip3(feat)
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret
            
        feat_coord = make_coord(feat_1x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat_1x.shape[0], 2, *feat_1x.shape[-2:])
        feat_coord_2x = make_coord(feat_2x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat_2x.shape[0], 2, *feat_2x.shape[-2:])
        feat_coord_4x = make_coord(feat_4x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat_4x.shape[0], 2, *feat_4x.shape[-2:])
        feat_coord_8x = make_coord(feat_8x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat_8x.shape[0], 2, *feat_8x.shape[-2:])
        
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        # 采样特征
        q_feat = F.grid_sample(
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
        # 采样坐标
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord2x = F.grid_sample(
            feat_coord_2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord4x = F.grid_sample(
            feat_coord_4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord8x = F.grid_sample(
            feat_coord_8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
                
        # space区域
        r = 2
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
        # 1, 1, r_area, 2
        space_delta_grids = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)
        spaces_coord = q_coord.unsqueeze(2) + space_delta_grids
        
        # ********* K & V ***********
        # K: b q spaces c 
        spaces_feat = F.grid_sample(
            feat_1x, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1).reshape(bs*q, -1, 64)       
        
        
        q_feat, _ = self.sp_attn(q_feat.reshape(bs*q, 1, 64), spaces_feat, spaces_feat)
        q_feat = q_feat.reshape(bs, q, 64)
        # -----方法一： 多尺度特征直接concat
        # inp = torch.cat([q_feat,q_feat2x,q_feat4x,q_feat8x], dim=-1)
        
        # -------方法二：多尺度特征transformer attention
        # inp = torch.stack([q_feat,q_feat2x,q_feat4x,q_feat8x], dim=-2)
        # inp = inp.reshape(bs*q,4,64)
        # inp = self.mult_attn(inp, inp, inp)[0]
        # inp = inp.reshape(bs,q,4*64)
        
       
        # rel_coord = coord - q_coord
        # rel_coord2x = coord - q_coord2x
        # rel_coord4x = coord - q_coord4x
        # rel_coord8x = coord - q_coord8x
        
        all_feat  = torch.cat([q_feat,q_feat2x,q_feat4x,q_feat8x], dim=-2).reshape(bs*q, 4, 64)
        # all_rel_coord = torch.cat([rel_coord, rel_coord2x, rel_coord4x, rel_coord8x], dim=-2).reshape(bs*q, 4, 2)
        
        # coords_embed = self.coord_projection(all_rel_coord)  # (batch_size, 4, d_model)
        # keys_values = all_feat + coords_embed  # (batch_size, 4, d_model) 当k和v
        
        # 使用多头注意力计算
        attn_output, _ = self.sc_attn(all_feat, all_feat, all_feat)
        inp = attn_output.reshape(bs, q, 64*4)
        
        # -------方法三：多尺度特征根据相对坐标插值，eg 双线性、双三次、距离指数attention
        # # 计算查询点与所有相对坐标点之间的距离
        # # 这里假设插值是在第 2 个维度上（即 4 的维度）
        # query_coord = torch.zeros(bs*q, 1, 2).cuda()  # 查询点相对坐标设为 (0, 0)
        # # 计算相似度得分，使用负的L2距离
        # distances = torch.cdist(query_coord, all_rel_coord, p=2)  # 形状为 (batchsize, 1, 4)
        # # 将距离转为负值表示相似度，并通过softmax归一化，得到注意力权重
        # attention_weights = F.softmax(-distances, dim=-1)  # 形状为 (batchsize, 1, 4)
        # # 对特征张量进行加权求和
        # interpolated_features = torch.bmm(attention_weights, all_feat)  # 形状为 (batchsize, 1, 64)
        # # 移除中间的维度，得到最终插值后的特征张量
        # inp = interpolated_features.squeeze(1).reshape(bs,q,64)  # 形状为 (batchsize, 64)
    
        if self.cell_decode:
            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]
            inp = torch.cat([inp, rel_cell], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        ret = pred
        # 残差链接
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)    
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

@register('liif-fpn-ca')
class LIIF_fpn_ca(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.c_attn = ChannelAttention(4*64)
        
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
            # skip 块
            self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip3 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            # HR->LR 对应4-》5快
            self.downsample_8_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            self.downsample_4_2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            self.downsample_2_1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
            
            imnet_in_dim = imnet_in_dim*4
            # self.mult_attn = nn.MultiheadAttention(64, 8, batch_first=True)
            
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
        # FPN
        feat_2 = self.upsample_1_2(feat) # 1->2
        feat_4 = self.upsample_2_4(feat_2) # 2->3
        feat_8x = self.upsample_4_8(feat_4) # 3->4
        
        feat_4x = self.downsample_8_4(feat_8x) + self.skip1(feat_4) # 4->3
        feat_2x = self.downsample_4_2(feat_4x) + self.skip2(feat_2) # 3->2
        feat_1x = self.downsample_2_1(feat_2x) + self.skip3(feat)
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
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
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        preds = []
        areas = []
        bs, q = coord.shape[:2]
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
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
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                # 多尺度特征
                inp = torch.cat([q_feat,q_feat2x,q_feat4x,q_feat8x], dim=-1)
                inp = self.c_attn(inp)*inp
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_coord],dim=-1)
                
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
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)    
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

@register('liif-FPN')
class LIIF_FPN(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        # self.c_attn = ChannelAttention(4*64)
        
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
                # LR->HR  对应1-》3快
            # self.upsample_1_2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
            # self.upsample_2_4 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
            # self.upsample_4_8 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
            
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
            
            # skip 块
            self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            self.skip3 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
            # HR->LR 对应4-》5快
            self.downsample_8_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 1, 2)
            self.downsample_4_2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1, 2)
            self.downsample_2_1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1, 2)
            
            imnet_in_dim = imnet_in_dim*4
            # self.mult_attn = nn.MultiheadAttention(64, 8, batch_first=True)
            
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
        # FPN
        feat_2 = self.upsample_1_2(feat) # 1->2
        feat_4 = self.upsample_2_4(feat_2) # 2->3
        feat_8x = self.upsample_4_8(feat_4) # 3->4
        
        feat_4x = self.downsample_8_4(feat_8x) + self.skip1(feat_4) # 4->3
        feat_2x = self.downsample_4_2(feat_4x) + self.skip2(feat_2) # 3->2
        feat_1x = self.downsample_2_1(feat_2x) + self.skip3(feat)
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
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
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        preds = []
        areas = []
        bs, q = coord.shape[:2]
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
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
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                # 多尺度特征
                inp = torch.cat([q_feat,q_feat2x,q_feat4x,q_feat8x], dim=-1)
                # inp = self.c_attn(inp)*inp
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([inp, rel_coord],dim=-1)
                
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
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)    
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

@register('liif-newbaseline')
class LIIF_newbaseline(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        
        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            num_heads = 8
            print(imnet_in_dim.shape)
            # 接受输入 B Seq V
            self.multi_attn = nn.MultiheadAttention(imnet_in_dim,num_heads,batch_first=True)
            self.pos2emd = nn.Linear(2, imnet_in_dim)
            
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None
            
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        
        B,C,H,W = feat.shape
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        coord_ = coord.clone()
        
        query = self.pos2emd(coord_) # seq: q
        
        k_feat = feat.view(B,C, -1).permute(0,2,1) # seq:H*W
            
        inp = self.multi_attn(query, k_feat, k_feat)

        bs, q = coord.shape[:2]
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
        
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)



 

@register('liif-fpn-sp-sc-attn-ours')
class LIIF_FPN_space_scale_atn_ours(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, pe_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        self.shuffle2x = nn.PixelShuffle(2)
        self.shuffle4x = nn.PixelShuffle(4)
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        # skip 块
        self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        
        # HR->LR 对应4-》5快
        self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        
        if imnet_spec is not None:
            if self.feat_unfold:
                imnet_in_dim = imnet_in_dim * 9 + int(imnet_in_dim*25/4) + int(imnet_in_dim*49/16)
            
            # 四个尺度（一个去查其他三个尺度）所以维度为3
            imnet_in_dim = imnet_in_dim
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
        # FPN
        feat_1 = self.upsample_1(feat) # 96
        feat_2 = self.upsample_2(feat_1) # 192
        feat_3 = self.upsample_3(feat_2) # 384
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2)
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1)
        
        feat = feat
        feat2x = feat_5
        feat4x = feat_4
        feat8x = feat_3
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
            feat2x = F.unfold(feat2x, 3, padding=1).view(
                feat2x.shape[0], feat2x.shape[1] * 9, feat2x.shape[2], feat2x.shape[3])
            
            feat4x = F.unfold(feat4x, 3, padding=1).view(
                feat4x.shape[0], feat4x.shape[1] * 9, feat4x.shape[2], feat4x.shape[3])
            
            feat8x = F.unfold(feat8x, 3, padding=1).view(
                feat8x.shape[0], feat8x.shape[1] * 9, feat8x.shape[2], feat8x.shape[3])
         
         
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        feat2x_coord = make_coord(feat2x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat2x.shape[0], 2, *feat2x.shape[-2:])
        
        feat4x_coord = make_coord(feat4x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat4x.shape[0], 2, *feat4x.shape[-2:])
        
        feat8x_coord = make_coord(feat8x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat8x.shape[0], 2, *feat8x.shape[-2:])
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
                
        q_coord_2x = F.grid_sample(
            feat2x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_4x = F.grid_sample(
            feat4x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_8x = F.grid_sample(
            feat8x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            
        # ********* Q ***********
        # b q 1 c -> 分多头 b q h 1 c
        q_feat = q_feat.unsqueeze(2).reshape(
            bs, q, 1, self.head, self.dim // self.head
        ).permute(0, 1, 3, 2, 4)
        # ********* K ***********
        # space上的cross attn 
        r = 3
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
        # 1, 1, r_area, 2
        delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)
        
        spaces_coord = q_coord.unsqueeze(2) + delta
        
        # 以q的近邻采样为k  feature: b q spaces c
        spaces_k_feat = F.grid_sample(
            feat, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        spaces = spaces_k_feat.shape[2]
        
        # b q spaces c -> 分多头b q spaces h c -> b q h c spaces
        spaces_k_feat = spaces_k_feat.reshape(
            bs, q, spaces, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        spaces_v_feat = spaces_k_feat.clone().permute(0, 1, 2, 4, 3) 
        
        # space上的 cross attention
        # LR_coord在multi_coord上的偏移
        # space_rel_coord = q_coord.unsqueeze(2) - spaces_coord
        # space_rel_coord[:, :, 0] *= feat.shape[-2]
        # space_rel_coord[:, :, 1] *= feat.shape[-1]
        # _,sp_pb = self.pe_encoder(space_rel_coord)
        
        
        # 加入位置编码 q k
        sp_q_pe = self.pe_encoder(q_coord.unsqueeze(2))[0].reshape(bs, q, 1, self.head, self.dim//self.head).permute(0, 1, 3, 2, 4)
        sp_k_pe = self.pe_encoder(spaces_coord)[0].reshape(bs, q, spaces, self.head, self.dim//self.head).permute(0, 1, 3, 4, 2)
        sp_v_pe = sp_k_pe.clone().permute(0, 1, 2, 4, 3)
        
        sp_attn = torch.matmul(torch.add(q_feat ,sp_q_pe), torch.add(spaces_k_feat, sp_k_pe)) / np.sqrt(self.dim // self.head)
        # sp_pb = sp_pb.permute(0,1,3,2).unsqueeze(2) # b q 扩展维度 1 scales
        
        sp_attn = F.softmax(sp_attn, dim=-1) 
        
        q_feat = torch.matmul(sp_attn,torch.add(spaces_v_feat,sp_v_pe)).reshape(
            bs, q ,1,self.head, self.dim // self.head
            ).permute(0, 1, 3, 2, 4)
        
        
        # scale上的cross attn 含有scale上的四个feat和coord    
        # 将2x 4x 8x特征一起做k
        #K: b q scales c  
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=2)
        scales = scales_k_feat.shape[2]
        # b q scales c -> 分多头 b q scales h c -> b q h c scales
        scales_k_feat = scales_k_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        
        # V:  b q h scales c 
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 2, 4, 3) 
       
        # ------------scale上的cross attn 含有scale上的四个feat和coord
        scales_coord = torch.stack([q_coord_2x, q_coord_4x, q_coord_8x], dim=2) # 多个高分辨坐标
        
        # LR_coord在multi_coord上的偏移
        # rel_coord = q_coord.unsqueeze(2) - scales_coord
        # rel_coord[:, :, 0] *= feat.shape[-2]
        # rel_coord[:, :, 1] *= feat.shape[-1]
        # multi-scale上的 cross attention
        # _,pb = self.pe_encoder(rel_coord)
        
        sc_k_pe = self.pe_encoder(scales_coord)[0].reshape(bs, q, scales, self.head, self.dim//self.head).permute(0, 1, 3, 4, 2)
        sc_v_pe = sc_k_pe.clone().permute(0, 1, 2, 4, 3)
        
        
        # b q head 1 scales
        sc_attn = torch.matmul(torch.add(q_feat ,sp_q_pe), torch.add(scales_k_feat, sc_k_pe)) / np.sqrt(self.dim // self.head)
    
        # pb = pb.permute(0,1,3,2).unsqueeze(2) # b q 扩展维度 1 scales
        sc_attn = F.softmax(sc_attn, dim=-1) # 加入pos-bias

        scales_v_feat = torch.matmul(sc_attn,torch.add(scales_v_feat,sc_v_pe)).reshape(bs, q, -1)
        
        # 相对坐标编码
        # rel_coord = coord_ - q_coord
        # rel_coord[:, :, 0] *= feat.shape[-2]
        # rel_coord[:, :, 1] *= feat.shape[-1]
        # rel_pe,_ = self.pe_encoder(rel_coord)
        # scales_v_feat = torch.add(scales_v_feat, rel_pe)
        
        if self.cell_decode:
            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]
            inp = torch.cat([scales_v_feat, rel_cell], dim=-1)
        
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-fpn-cell-sc-attn')
class LIIF_FPN_cell_scale_atn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, pe_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        self.shuffle2x = nn.PixelShuffle(2)
        self.shuffle4x = nn.PixelShuffle(4)
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        # self.upsample_1 = nn.Sequential(
        #     nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
        #     nn.PixelShuffle(2)
        # )
        # self.upsample_2 = nn.Sequential(
        #     nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
        #     nn.PixelShuffle(2)
        # )
        # self.upsample_3 = nn.Sequential(
        #     nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
        #     nn.PixelShuffle(2)
        # )
        
        self.upsample_1 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        self.upsample_2 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        self.upsample_3 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        
        
        # skip 块
        self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        
        # HR->LR 对应4-》5快
        self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
        
        if imnet_spec is not None:
            # 四个尺度（一个去查其他三个尺度）所以维度为3
            imnet_in_dim = imnet_in_dim
            # if self.cell_decode:
            #     imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        # --------- FPN -----------
        feat_1 = self.upsample_1(feat) # 96
        feat_2 = self.upsample_2(feat_1) # 192
        feat_3 = self.upsample_3(feat_2) # 384
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2)
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1)
        
        feat = feat
        feat2x = feat_5
        feat4x = feat_4
        feat8x = feat_3
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        feat2x_coord = make_coord(feat2x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat2x.shape[0], 2, *feat2x.shape[-2:])
        
        feat4x_coord = make_coord(feat4x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat4x.shape[0], 2, *feat4x.shape[-2:])
        
        feat8x_coord = make_coord(feat8x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat8x.shape[0], 2, *feat8x.shape[-2:])
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        # 多尺度feature及其对应的LR坐标
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_2x = F.grid_sample(
            feat2x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_4x = F.grid_sample(
            feat4x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_8x = F.grid_sample(
            feat8x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            
        # ********* Q ***********
        # b q 1 c -> 分多头 b q h 1 c
        q_feat = q_feat.unsqueeze(2).reshape(
            bs, q, 1, self.head, self.dim // self.head
        ).permute(0, 1, 3, 2, 4)
        
        # ********* K ***********
        grids = []
        for i in range(bs):
            rh = cell[i,0,0] / 2.
            rw = cell[i,0,1] / 2.
            dh = torch.tensor([-rh,0,rh]).cuda()
            dw = torch.tensor([-rw,0,rw]).cuda()
            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, -1, 2)
            grids.append(delta)
        delta_grids = torch.cat(grids, dim=0).unsqueeze(1)
        
        spaces_coord = q_coord.unsqueeze(2) + delta_grids
        
        # 以q的近邻采样为k  feature: b q spaces c
        spaces_k_feat = F.grid_sample(
            feat, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        
        # K: b q spaces c -> 分多头b q spaces h c -> b q h c spaces
        spaces = spaces_k_feat.shape[2] # 近邻数
        spaces_k_feat = spaces_k_feat.reshape(
            bs, q, spaces, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        
        # ********* V ***********
        # V:  b q h spaces c 
        spaces_v_feat = spaces_k_feat.clone().permute(0, 1, 2, 4, 3) 
        
        sp_attn = torch.matmul(q_feat, spaces_k_feat) / np.sqrt(self.dim // self.head)
        sp_attn = self.attend(sp_attn)
        sp_attn = self.dropout(sp_attn)
        
        # 经过cell-attention之后的q
        q_feat = torch.matmul(sp_attn,spaces_v_feat)
        
        # ********* K ***********
        # scale上的cross attn 含有scale上的四个feat和coord    
        # K: b q scales c  
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=2)
        
        # K: b q scales c -> 分多头 b q scales h c -> b q h c scales
        scales = scales_k_feat.shape[2]
        scales_k_feat = scales_k_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        
        # ********* V ***********
        # V:  b q h scales c 
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 2, 4, 3) 
       
        sc_attn = torch.matmul(q_feat, scales_k_feat) / np.sqrt(self.dim // self.head)
        sc_attn = self.attend(sc_attn)
        sc_attn = self.dropout(sc_attn)
        
        inp = torch.matmul(sc_attn,scales_v_feat).reshape(bs*q, -1)
        
        
        
        pred = self.imnet(inp).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-fpn-sp-sc-cell-attn')
class LIIF_FPN_space_scale_cell_atn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, pe_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
    
        self.upsample_1 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        self.upsample_2 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        self.upsample_3 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        
        # skip 块
        self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        
        # HR->LR 对应4-》5快
        self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
        
        if imnet_spec is not None:
            # cell的点有9个
            imnet_in_dim = imnet_in_dim
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        
        
        # FPN
        feat_1 = self.upsample_1(feat) # 96
        feat_2 = self.upsample_2(feat_1) # 192
        feat_3 = self.upsample_3(feat_2) # 384
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2)
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1)
        
        feat = feat
        feat2x = feat_5
        feat4x = feat_4
        feat8x = feat_3
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
            feat2x = F.unfold(feat2x, 3, padding=1).view(
                feat2x.shape[0], feat2x.shape[1] * 9, feat2x.shape[2], feat2x.shape[3])
            
            feat4x = F.unfold(feat4x, 3, padding=1).view(
                feat4x.shape[0], feat4x.shape[1] * 9, feat4x.shape[2], feat4x.shape[3])
            
            feat8x = F.unfold(feat8x, 3, padding=1).view(
                feat8x.shape[0], feat8x.shape[1] * 9, feat8x.shape[2], feat8x.shape[3])
         
         
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        feat2x_coord = make_coord(feat2x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat2x.shape[0], 2, *feat2x.shape[-2:])
        
        feat4x_coord = make_coord(feat4x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat4x.shape[0], 2, *feat4x.shape[-2:])
        
        feat8x_coord = make_coord(feat8x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat8x.shape[0], 2, *feat8x.shape[-2:])
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        # b q 1 c -> 分多头b q 1 h c -> b q h 1 c
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(2),
            mode='nearest', align_corners=False) \
            .permute(0, 2, 3, 1)
        q_feat = q_feat.reshape(bs, q, 1, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
                
        # q_coord_2x = F.grid_sample(
        #     feat2x_coord, coord_.flip(-1).unsqueeze(1),
        #     mode='nearest', align_corners=False)[:, :, 0, :] \
        #         .permute(0, 2, 1)
        # q_coord_4x = F.grid_sample(
        #     feat4x_coord, coord_.flip(-1).unsqueeze(1),
        #     mode='nearest', align_corners=False)[:, :, 0, :] \
        #         .permute(0, 2, 1)
        # q_coord_8x = F.grid_sample(
        #     feat8x_coord, coord_.flip(-1).unsqueeze(1),
        #     mode='nearest', align_corners=False)[:, :, 0, :] \
        #         .permute(0, 2, 1)
            
        # cell区域
        grids = []
        for i in range(bs):
            rh = cell[i,0,0] / 2.
            rw = cell[i,0,1] / 2.
            dh = torch.tensor([-rh,0,rh]).cuda()
            dw = torch.tensor([-rw,0,rw]).cuda()
            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, -1, 2)
            grids.append(delta)
        delta_grids = torch.cat(grids, dim=0).unsqueeze(1)
        cell_spaces_coord = q_coord.unsqueeze(2) + delta_grids
        
        # ********* Q ***********
        # b q cell c -> 分多头b q cell h c -> b q h  cell c
        cell_q_feat = F.grid_sample(
            feat, cell_spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        cell = cell_q_feat.shape[2]
        cell_q_feat = cell_q_feat.reshape(
            bs, q, cell, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
        # space区域
        r = 2
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
        # 1, 1, r_area, 2
        space_delta_grids = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)
        spaces_coord = q_coord.unsqueeze(2) + space_delta_grids
        
        # ********* K & V ***********
        # K: b q spaces c -> 分多头b q spaces h c -> b q h c spaces
        spaces_k_feat = F.grid_sample(
            feat, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        spaces = spaces_k_feat.shape[2]
        spaces_k_feat = spaces_k_feat.reshape(
            bs, q, spaces, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        spaces_v_feat = spaces_k_feat.clone().permute(0, 1, 2, 4, 3) 
        
        # cell 对 space
        spaces_sp_attn = torch.matmul(cell_q_feat, spaces_k_feat) / np.sqrt(self.dim // self.head)
        spaces_sp_attn = self.attend(spaces_sp_attn)
        spaces_sp_attn = self.dropout(spaces_sp_attn)
        cell_q_feat = torch.matmul(spaces_sp_attn, spaces_v_feat)
        
        
        
        # ********* K & V ***********
        # K: b q scales c -> 分多头 b q scales h c -> b q h c scales
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=2)
        scales = scales_k_feat.shape[2]
        scales_k_feat = scales_k_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 2, 4, 3) 
       
        # cell 对 scale 
        sc_attn = torch.matmul(cell_q_feat , scales_k_feat) / np.sqrt(self.dim // self.head)
        sc_attn = self.attend(sc_attn)
        sc_attn = self.dropout(sc_attn)
        cell_k_feat = torch.matmul(sc_attn,scales_v_feat)
        
        # 1 对 cell
        cell_v_feat = cell_k_feat.clone() # b q h 4 c
        cell_k_feat = cell_k_feat.permute(0,1,2,4,3) # b q h c 4
        
        # cell 对 cell 的自注意力
        cell_attn = torch.matmul(q_feat, cell_k_feat) / np.sqrt(self.dim // self.head)
        cell_attn = self.attend(cell_attn)
        cell_attn = self.dropout(cell_attn)
        inp = torch.matmul(cell_attn, cell_v_feat)
        
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-fpn-sp-cell-sc-attn')
class LIIF_FPN_space_cell_scale_attn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=False):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        # self.upsample_1 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        # self.upsample_2 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        # self.upsample_3 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        
        # skip 块
        self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        
        # HR->LR 对应4-》5快
        self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        
        self.attend = nn.Softmax(dim = -1)
        
        if imnet_spec is not None:
            # cell的点有9个
            imnet_in_dim = imnet_in_dim + 2
            
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
            
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        # FPN
        feat_1 = self.upsample_1(feat) # 1->2
        feat_2 = self.upsample_2(feat_1) # 2->3
        feat_3 = self.upsample_3(feat_2) # 3->4
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2) # 4->3
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1) # 3->2
        
        feat = feat
        feat2x = feat_5
        feat4x = feat_4
        feat8x = feat_3
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
            feat2x = F.unfold(feat2x, 3, padding=1).view(
                feat2x.shape[0], feat2x.shape[1] * 9, feat2x.shape[2], feat2x.shape[3])
            
            feat4x = F.unfold(feat4x, 3, padding=1).view(
                feat4x.shape[0], feat4x.shape[1] * 9, feat4x.shape[2], feat4x.shape[3])
            
            feat8x = F.unfold(feat8x, 3, padding=1).view(
                feat8x.shape[0], feat8x.shape[1] * 9, feat8x.shape[2], feat8x.shape[3])
         
         
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        # b q 1 c -> 分多头b q 1 h c -> b q h 1 c
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(2),
            mode='nearest', align_corners=False) \
            .permute(0, 2, 3, 1)
        q_feat = q_feat.reshape(bs, q, 1, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
                
        # cell区域
        grids = []
        for i in range(bs):
            rh = cell[i,0,0] / 2.
            rw = cell[i,0,1] / 2.
            dh = torch.tensor([-rh,0,rh]).cuda()
            dw = torch.tensor([-rw,0,rw]).cuda()
            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, -1, 2)
            grids.append(delta)
        delta_grids = torch.cat(grids, dim=0).unsqueeze(1)
        cell_spaces_coord = q_coord.unsqueeze(2) + delta_grids
        
        # ********* Q ***********
        # b q cell c -> 分多头b q cell h c -> b q h  cell c
        cell_q_feat = F.grid_sample(
            feat, cell_spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        cell = cell_q_feat.shape[2]
        cell_q_feat = cell_q_feat.reshape(
            bs, q, cell, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
        # space区域
        r = 2
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
        # 1, 1, r_area, 2
        space_delta_grids = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)
        spaces_coord = q_coord.unsqueeze(2) + space_delta_grids
        
        # ********* K & V ***********
        # K: b q spaces c -> 分多头b q spaces h c -> b q h c spaces
        spaces_k_feat = F.grid_sample(
            feat, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        spaces = spaces_k_feat.shape[2]
        spaces_k_feat = spaces_k_feat.reshape(
            bs, q, spaces, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        spaces_v_feat = spaces_k_feat.clone().permute(0, 1, 2, 4, 3) 
        
        # cell 对 space
        spaces_sp_attn = torch.matmul(cell_q_feat, spaces_k_feat) / np.sqrt(self.dim // self.head)
        spaces_sp_attn = self.attend(spaces_sp_attn)
        cell_k_feat = torch.matmul(spaces_sp_attn, spaces_v_feat)
        
        # 1 对 cell
        cell_v_feat = cell_k_feat.clone() # b q h 4 c
        cell_k_feat = cell_k_feat.permute(0,1,2,4,3) # b q h c 4
         # cell 对 cell 的自注意力
        cell_attn = torch.matmul(q_feat, cell_k_feat) / np.sqrt(self.dim // self.head)
        cell_attn = self.attend(cell_attn)
        q_feat = torch.matmul(cell_attn, cell_v_feat)
        
        
        # ********* K & V ***********
        # K: b q scales c -> 分多头 b q scales h c -> b q h c scales
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=2)
        scales = scales_k_feat.shape[2]
        scales_k_feat = scales_k_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 2, 4, 3) 
       
        # cell 对 scale 
        sc_attn = torch.matmul(q_feat , scales_k_feat) / np.sqrt(self.dim // self.head)
        sc_attn = self.attend(sc_attn)
        inp = torch.matmul(sc_attn,scales_v_feat).view(bs, q, -1)
        
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        inp = torch.cat([inp, rel_coord], dim=-1)


        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-fpn-sp-sc-attn')
class LIIF_FPN_space_scale_attn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        # self.upsample_1 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        # self.upsample_2 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        # self.upsample_3 = nn.Upsample(scale_factor=2,mode='bicubic',align_corners=False)
        
        # skip 块
        self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        
        # HR->LR 对应4-》5快
        self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        
        self.attend = nn.Softmax(dim = -1)
        
        if imnet_spec is not None:
            imnet_in_dim += 2
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
        # FPN
        feat_1 = self.upsample_1(feat) # 1->2
        feat_2 = self.upsample_2(feat_1) # 2->3
        feat_3 = self.upsample_3(feat_2) # 3->4
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2) # 4->3
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1) # 3->2
        
        feat = feat
        feat2x = feat_5
        feat4x = feat_4
        feat8x = feat_3
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
            feat2x = F.unfold(feat2x, 3, padding=1).view(
                feat2x.shape[0], feat2x.shape[1] * 9, feat2x.shape[2], feat2x.shape[3])
            
            feat4x = F.unfold(feat4x, 3, padding=1).view(
                feat4x.shape[0], feat4x.shape[1] * 9, feat4x.shape[2], feat4x.shape[3])
            
            feat8x = F.unfold(feat8x, 3, padding=1).view(
                feat8x.shape[0], feat8x.shape[1] * 9, feat8x.shape[2], feat8x.shape[3])
         
         
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        # b q 1 c -> 分多头b q 1 h c -> b q h 1 c
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(2),
            mode='nearest', align_corners=False) \
            .permute(0, 2, 3, 1)
        q_feat = q_feat.reshape(bs, q, 1, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
                
        # space区域
        r = 2
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
        # 1, 1, r_area, 2
        space_delta_grids = torch.stack(torch.meshgrid(dh, dw), dim=-1).view(1, 1, -1, 2)
        spaces_coord = q_coord.unsqueeze(2) + space_delta_grids
        
        # ********* K & V ***********
        # K: b q spaces c -> 分多头b q spaces h c -> b q h c spaces
        spaces_k_feat = F.grid_sample(
            feat, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        spaces = spaces_k_feat.shape[2]
        spaces_k_feat = spaces_k_feat.reshape(
            bs, q, spaces, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        spaces_v_feat = spaces_k_feat.clone().permute(0, 1, 2, 4, 3) 
        
        # cell 对 space
        spaces_sp_attn = torch.matmul(q_feat, spaces_k_feat) / np.sqrt(self.dim // self.head)
        spaces_sp_attn = self.attend(spaces_sp_attn)
        q_feat = torch.matmul(spaces_sp_attn, spaces_v_feat)
        
        # ********* K & V **********
        # K: b q scales c -> 分多头 b q scales h c -> b q h c scales
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=2)
        scales = scales_k_feat.shape[2]
        scales_k_feat = scales_k_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 2, 4, 3) 
       
        # cell 对 scale 
        sc_attn = torch.matmul(q_feat , scales_k_feat) / np.sqrt(self.dim // self.head)
        sc_attn = self.attend(sc_attn)
        inp = torch.matmul(sc_attn,scales_v_feat).view(bs, q, -1)
        
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        inp = torch.cat([inp, rel_coord], dim=-1)

        if self.cell_decode:
            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]
            inp = torch.cat([inp, rel_cell], dim=-1)
                
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-fpn-sp-sc-cell-attn-dropout')
class LIIF_FPN_space_scale_cell_atn_drop(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, pe_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        
        self.feat_dropout = nn.Dropout(0.5)
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
    
        # skip 块
        self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        
        # HR->LR 对应4-》5快
        self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
        
        if imnet_spec is not None:
            # cell的点有9个
            imnet_in_dim = imnet_in_dim
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat_dropout(self.feat)
        
        # FPN
        feat_1 = self.upsample_1(feat) # 96
        feat_2 = self.upsample_2(feat_1) # 192
        feat_3 = self.upsample_3(feat_2) # 384
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2)
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1)
        
        feat = feat
        feat2x = feat_5
        feat4x = feat_4
        feat8x = feat_3
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
            feat2x = F.unfold(feat2x, 3, padding=1).view(
                feat2x.shape[0], feat2x.shape[1] * 9, feat2x.shape[2], feat2x.shape[3])
            
            feat4x = F.unfold(feat4x, 3, padding=1).view(
                feat4x.shape[0], feat4x.shape[1] * 9, feat4x.shape[2], feat4x.shape[3])
            
            feat8x = F.unfold(feat8x, 3, padding=1).view(
                feat8x.shape[0], feat8x.shape[1] * 9, feat8x.shape[2], feat8x.shape[3])
         
         
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        feat2x_coord = make_coord(feat2x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat2x.shape[0], 2, *feat2x.shape[-2:])
        
        feat4x_coord = make_coord(feat4x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat4x.shape[0], 2, *feat4x.shape[-2:])
        
        feat8x_coord = make_coord(feat8x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat8x.shape[0], 2, *feat8x.shape[-2:])
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        # b q 1 c -> 分多头b q 1 h c -> b q h 1 c
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(2),
            mode='nearest', align_corners=False) \
            .permute(0, 2, 3, 1)
        q_feat = q_feat.reshape(bs, q, 1, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
                
        # q_coord_2x = F.grid_sample(
        #     feat2x_coord, coord_.flip(-1).unsqueeze(1),
        #     mode='nearest', align_corners=False)[:, :, 0, :] \
        #         .permute(0, 2, 1)
        # q_coord_4x = F.grid_sample(
        #     feat4x_coord, coord_.flip(-1).unsqueeze(1),
        #     mode='nearest', align_corners=False)[:, :, 0, :] \
        #         .permute(0, 2, 1)
        # q_coord_8x = F.grid_sample(
        #     feat8x_coord, coord_.flip(-1).unsqueeze(1),
        #     mode='nearest', align_corners=False)[:, :, 0, :] \
        #         .permute(0, 2, 1)
            
        # cell区域
        grids = []
        for i in range(bs):
            rh = cell[i,0,0] / 2.
            rw = cell[i,0,1] / 2.
            dh = torch.tensor([-rh,0,rh]).cuda()
            dw = torch.tensor([-rw,0,rw]).cuda()
            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, -1, 2)
            grids.append(delta)
        delta_grids = torch.cat(grids, dim=0).unsqueeze(1)
        cell_spaces_coord = q_coord.unsqueeze(2) + delta_grids
        
        # ********* Q ***********
        # b q cell c -> 分多头b q cell h c -> b q h  cell c
        cell_q_feat = F.grid_sample(
            feat, cell_spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        cell = cell_q_feat.shape[2]
        cell_q_feat = cell_q_feat.reshape(
            bs, q, cell, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
        # space区域
        r = 2
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
        # 1, 1, r_area, 2
        space_delta_grids = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)
        spaces_coord = q_coord.unsqueeze(2) + space_delta_grids
        
        # ********* K & V ***********
        # K: b q spaces c -> 分多头b q spaces h c -> b q h c spaces
        spaces_k_feat = F.grid_sample(
            feat, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        spaces = spaces_k_feat.shape[2]
        spaces_k_feat = spaces_k_feat.reshape(
            bs, q, spaces, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        spaces_v_feat = spaces_k_feat.clone().permute(0, 1, 2, 4, 3) 
        
        # cell 对 space
        spaces_sp_attn = torch.matmul(cell_q_feat, spaces_k_feat) / np.sqrt(self.dim // self.head)
        spaces_sp_attn = self.attend(spaces_sp_attn)
        spaces_sp_attn = self.dropout(spaces_sp_attn)
        cell_q_feat = torch.matmul(spaces_sp_attn, spaces_v_feat)
        
        
        
        # ********* K & V ***********
        # K: b q scales c -> 分多头 b q scales h c -> b q h c scales
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=2)
        scales = scales_k_feat.shape[2]
        scales_k_feat = scales_k_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 2, 4, 3) 
       
        # cell 对 scale 
        sc_attn = torch.matmul(cell_q_feat , scales_k_feat) / np.sqrt(self.dim // self.head)
        sc_attn = self.attend(sc_attn)
        sc_attn = self.dropout(sc_attn)
        cell_k_feat = torch.matmul(sc_attn,scales_v_feat)
        
        # 1 对 cell
        cell_v_feat = cell_k_feat.clone() # b q h 4 c
        cell_k_feat = cell_k_feat.permute(0,1,2,4,3) # b q h c 4
        
        # cell 对 cell 的自注意力
        cell_attn = torch.matmul(q_feat, cell_k_feat) / np.sqrt(self.dim // self.head)
        cell_attn = self.attend(cell_attn)
        cell_attn = self.dropout(cell_attn)
        inp = torch.matmul(cell_attn, cell_v_feat)
        
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-fpn-sp-sc-cell-attn-mask')
class LIIF_FPN_space_scale_cell_atn_mask(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, pe_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        self.shuffle2x = nn.PixelShuffle(2)
        self.shuffle4x = nn.PixelShuffle(4)
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bicubic',align_corners=False)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bicubic',align_corners=False)
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bicubic',align_corners=False)
    
        # skip 块
        self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        
        # HR->LR 对应4-》5快
        self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
        
        if imnet_spec is not None:
            # cell的点有9个
            imnet_in_dim = imnet_in_dim
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp, mask=None):
        self.inp = inp
        self.mask = mask
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        # FPN
        feat_1 = self.upsample_1(feat) # 96
        feat_2 = self.upsample_2(feat_1) # 192
        feat_3 = self.upsample_3(feat_2) # 384
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2)
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1)
        
        feat = feat
        feat2x = feat_5
        feat4x = feat_4
        feat8x = feat_3
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        # b q 1 c -> 分多头b q 1 h c -> b q h 1 c
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(2),
            mode='nearest', align_corners=False).permute(0, 2, 3, 1)
        q_feat = q_feat.reshape(bs, q, 1, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
       
        
        
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        
        # cell区域
        grids = []
        for i in range(bs):
            rh = cell[i,0,0] / 2.
            rw = cell[i,0,1] / 2.
            dh = torch.tensor([-rh,0,rh]).cuda()
            dw = torch.tensor([-rw,0,rw]).cuda()
            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, -1, 2)
            grids.append(delta)
        delta_grids = torch.cat(grids, dim=0).unsqueeze(1)
        cell_spaces_coord = q_coord.unsqueeze(2) + delta_grids
        
        # ********* Q ***********
        # b q cell c -> 分多头b q cell h c -> b q h cell c
        cell_q_feat = F.grid_sample(
            feat, cell_spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        cell = cell_q_feat.shape[2]
        cell_q_feat = cell_q_feat.reshape(
            bs, q, cell, self.head, self.dim // self.head).permute(0, 1, 3, 2, 4)
        
        # space区域
        r = 2
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
        # 1, 1, r_area, 2
        space_delta_grids = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)
        spaces_coord = q_coord.unsqueeze(2) + space_delta_grids
        
        # ********* K & V ***********
        # K: b q spaces c -> 分多头b q spaces h c -> b q h c spaces
        spaces_k_feat = F.grid_sample(
            feat, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        spaces = spaces_k_feat.shape[2]
        spaces_k_feat = spaces_k_feat.reshape(
            bs, q, spaces, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        
        
        # print(spaces_coord.shape)
        if self.mask is not None:
            # 采样mask: b q spaces 1
            atten_mask = F.grid_sample(
                self.mask, spaces_coord.flip(-1),
                mode='nearest', align_corners=False).permute(0, 2, 3, 1)

        
        # V:  b q h spaces c 
        spaces_v_feat = spaces_k_feat.clone().permute(0, 1, 2, 4, 3) 
        
        # cell 对 space
        spaces_sp_attn = torch.matmul(cell_q_feat, spaces_k_feat) / np.sqrt(self.dim // self.head)

       

        spaces_sp_attn = self.attend(spaces_sp_attn)
        spaces_sp_attn = self.dropout(spaces_sp_attn)

        
        if self.mask is not None:
            
            atten_mask = atten_mask.unsqueeze(2).unsqueeze(3).squeeze(-1)
            spaces_sp_attn = spaces_sp_attn * atten_mask
        # attn: b q h cell spaces

        cell_q_feat = torch.matmul(spaces_sp_attn, spaces_v_feat)
        
        
        
        # ********* K & V ***********
        # K: b q scales c -> 分多头 b q scales h c -> b q h c scales
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=2)
        scales = scales_k_feat.shape[2]
        scales_k_feat = scales_k_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 2, 4, 3) 
       
        # cell 对 scale 
        sc_attn = torch.matmul(cell_q_feat , scales_k_feat) / np.sqrt(self.dim // self.head)
        sc_attn = self.attend(sc_attn)
        sc_attn = self.dropout(sc_attn)
        cell_k_feat = torch.matmul(sc_attn,scales_v_feat)
        
        
        
        # 1 对 cell
        cell_v_feat = cell_k_feat.clone() # b q h 4 c
        cell_k_feat = cell_k_feat.permute(0,1,2,4,3) # b q h c 4
        
        # cell 对 cell 的自注意力
        cell_attn = torch.matmul(q_feat, cell_k_feat) / np.sqrt(self.dim // self.head)
        cell_attn = self.attend(cell_attn)
        cell_attn = self.dropout(cell_attn)
        inp = torch.matmul(cell_attn, cell_v_feat)
        
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)

        
        return pred

    def forward(self, inp, coord, cell, mask=None):
        self.gen_feat(inp,mask)
        return self.query_rgb(coord, cell)




@register('liif-fpn-sc-sp-attn')
class LIIF_FPN_scale_space_attn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, pe_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        self.shuffle2x = nn.PixelShuffle(2)
        self.shuffle4x = nn.PixelShuffle(4)
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        # skip 块
        self.skip1 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        self.skip2 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 1)
        
        # HR->LR 对应4-》5快
        self.downsample_4= nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        self.downsample_5 = nn.Conv2d(imnet_in_dim, imnet_in_dim, 3, 2, 1)
        
        if imnet_spec is not None:
            if self.feat_unfold:
                imnet_in_dim = imnet_in_dim * 9 + int(imnet_in_dim*25/4) + int(imnet_in_dim*49/16)
            
            # 四个尺度（一个去查其他三个尺度）所以维度为3
            imnet_in_dim = imnet_in_dim
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
        # FPN
        feat_1 = self.upsample_1(feat) # 96
        feat_2 = self.upsample_2(feat_1) # 192
        feat_3 = self.upsample_3(feat_2) # 384
        
        feat_4 = self.downsample_4(feat_3) + self.skip1(feat_2)
        feat_5 = self.downsample_5(feat_4) + self.skip2(feat_1)
        
        feat = feat
        feat2x = feat_5
        feat4x = feat_4
        feat8x = feat_3
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            
            feat2x = F.unfold(feat2x, 3, padding=1).view(
                feat2x.shape[0], feat2x.shape[1] * 9, feat2x.shape[2], feat2x.shape[3])
            
            feat4x = F.unfold(feat4x, 3, padding=1).view(
                feat4x.shape[0], feat4x.shape[1] * 9, feat4x.shape[2], feat4x.shape[3])
            
            feat8x = F.unfold(feat8x, 3, padding=1).view(
                feat8x.shape[0], feat8x.shape[1] * 9, feat8x.shape[2], feat8x.shape[3])
         
         
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
            
        feat2x_coord = make_coord(feat2x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat2x.shape[0], 2, *feat2x.shape[-2:])
        
        feat4x_coord = make_coord(feat4x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat4x.shape[0], 2, *feat4x.shape[-2:])
        
        feat8x_coord = make_coord(feat8x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat8x.shape[0], 2, *feat8x.shape[-2:])
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
                
        q_coord_2x = F.grid_sample(
            feat2x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_4x = F.grid_sample(
            feat4x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_8x = F.grid_sample(
            feat8x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            
        # ********* Q ***********
        # b q 1 c -> 分多头 b q h 1 c
        q_feat = q_feat.unsqueeze(2).reshape(
            bs, q, 1, self.head, self.dim // self.head
        ).permute(0, 1, 3, 2, 4)
        # ********* K ***********
        # space上的cross attn 
        r = 3
        dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
        dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
        # 1, 1, r_area, 2
        delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)
        spaces_coord = q_coord.unsqueeze(2) + delta
        
        
        
        # ------------- scale上的cross attn ----------   
        # 将2x 4x 8x特征一起做k
        #K: b q scales c  
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=2)
        scales = scales_k_feat.shape[2]
        # b q scales c -> 分多头 b q scales h c -> b q h c scales
        scales_k_feat = scales_k_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        
        # V:  b q h scales c 
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 2, 4, 3) 
       
        # ------------scale上的cross attn 含有scale上的四个feat和coord
        scales_coord = torch.stack([q_coord_2x, q_coord_4x, q_coord_8x], dim=2) # 多个高分辨坐标
        
        # LR_coord在multi_coord上的偏移
        rel_coord = q_coord.unsqueeze(2) - scales_coord
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        
        # multi-scale上的 cross attention
        _,pb = self.pe_encoder(rel_coord)
        
        # b q head 1 scales
        attn = torch.matmul(q_feat, scales_k_feat) / np.sqrt(self.dim // self.head)
    
        pb = pb.permute(0,1,3,2).unsqueeze(2) # b q 扩展维度 1 scales
        attn = F.softmax(torch.add(attn, pb), dim=-1) # 加入pos-bias

        q_feat = torch.matmul(attn,scales_v_feat).reshape(
            bs, q ,1,self.head, self.dim // self.head
            ).permute(0, 1, 3, 2, 4)
        
        
        
        
        
        # 以q的近邻采样为k  feature: b q spaces c
        spaces_k_feat = F.grid_sample(
            feat, spaces_coord.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)
        spaces = spaces_k_feat.shape[2]
        # b q spaces c -> 分多头b q spaces h c -> b q h c spaces
        spaces_k_feat = spaces_k_feat.reshape(
            bs, q, spaces, self.head, self.dim // self.head).permute(0, 1, 3, 4, 2)
        # V:  b q h scales c 
        spaces_v_feat = spaces_k_feat.clone().permute(0, 1, 2, 4, 3) 
        
        # space上的 cross attention
        # LR_coord在multi_coord上的偏移
        space_rel_coord = q_coord.unsqueeze(2) - spaces_coord
        space_rel_coord[:, :, 0] *= feat.shape[-2]
        space_rel_coord[:, :, 1] *= feat.shape[-1]
        _,sp_pb = self.pe_encoder(space_rel_coord)
        
        sp_attn = torch.matmul(q_feat, spaces_k_feat) / np.sqrt(self.dim // self.head)
        sp_pb = sp_pb.permute(0,1,3,2).unsqueeze(2) # b q 扩展维度 1 scales
        sp_attn = F.softmax(torch.add(sp_attn, sp_pb), dim=-1) # 加入pos-bias
        
        spaces_v_feat = torch.matmul(sp_attn,spaces_v_feat).reshape(bs, q, -1)
        
        
        if self.cell_decode:
            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]
            inp = torch.cat([spaces_v_feat, rel_cell], dim=-1)
        
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-fft-attn')
class LIIF_FFT_atn(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, pe_spec=None,head=8,
                 local_ensemble=False, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        self.shuffle2x = nn.PixelShuffle(2)
        self.shuffle4x = nn.PixelShuffle(4)
        
        imnet_in_dim = self.encoder.out_dim
        self.dim = imnet_in_dim
        # LR->HR  对应1-》3快
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_3 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        if imnet_spec is not None:
            
            # 四个尺度（一个去查其他三个尺度）所以维度为3
            imnet_in_dim = imnet_in_dim * 3
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
        # FPN
        feat_1 = self.upsample_1(feat) # 96x96
        feat_2 = self.upsample_2(feat_1) # 192x192
        feat_3 = self.upsample_3(feat_2) # 384x384
        
        feat_fft = torch.fft.fft2(feat,dim=(-2,-1))
        feat_fft2x = torch.fft.fft2(feat_1, dim=(-2, -1))
        feat_fft4x = torch.fft.fft2(feat_2, dim=(-2, -1))
        feat_fft8x = torch.fft.fft2(feat_3, dim=(-2, -1))
        
        h1,w1 = feat_fft.shape[-2:]
        h2,w2 = feat_fft2x.shape[-2:]
        h3,w3 = feat_fft4x.shape[-2:]
        h4,w4 = feat_fft8x.shape[-2:]
        
        feat_fft_shift = torch.fft.fftshift(feat_fft, dim=(-2, -1))
        feat_fft2x_shift = torch.fft.fftshift(feat_fft2x, dim=(-2, -1))
        feat_fft4x_shift = torch.fft.fftshift(feat_fft4x, dim=(-2, -1))
        feat_fft8x_shift = torch.fft.fftshift(feat_fft8x, dim=(-2, -1))
        # 高频的feature备份
        feat_fft2x_shift_ = feat_fft2x_shift.clone()
        feat_fft4x_shift_ = feat_fft4x_shift.clone()
        feat_fft8x_shift_ = feat_fft8x_shift.clone()
        
        # 层层嵌套中间套前面得到多尺度feature,作为q
        feat_fft2x_shift[:,:,((h2-h1) // 2): ((h2-h1) // 2) + h1 , ((w2-w1) // 2): ((w2-w1) // 2) + w1 ] = feat_fft_shift
        feat_fft4x_shift[:,:,((h3-h2) // 2): ((h3-h2) // 2) + h2 , ((w3-w2) // 2): ((w3-w2) // 2) + w2 ] = feat_fft2x_shift
        feat_fft8x_shift[:,:,((h4-h3) // 2): ((h4-h3) // 2) + h3 , ((w4-w3) // 2): ((w4-w3) // 2) + w3 ] = feat_fft4x_shift
        
        feat = torch.real(torch.fft.ifft2(feat_fft8x_shift)) # 转为实数feature,384x384 
        
        # 多尺度feature
        feat2x = torch.real(torch.fft.ifft2(feat_fft2x_shift_)) # 48x48
        feat4x = torch.real(torch.fft.ifft2(feat_fft4x_shift_)) # 96x96
        feat8x = torch.real(torch.fft.ifft2(feat_fft8x_shift_)) # 192x192
        
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        feat2x_coord = make_coord(feat2x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat2x.shape[0], 2, *feat2x.shape[-2:])
        
        feat4x_coord = make_coord(feat4x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat4x.shape[0], 2, *feat4x.shape[-2:])
        
        feat8x_coord = make_coord(feat8x.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
                .unsqueeze(0).expand(feat8x.shape[0], 2, *feat8x.shape[-2:])
            
        bs, q = coord.shape[:2]
        coord_ = coord.clone()
        
        # 原feature
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
            
        q_feat_2x = F.grid_sample(
            feat2x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1) 
        q_feat_4x = F.grid_sample(
            feat4x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)      
        q_feat_8x = F.grid_sample(
            feat8x, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
                
        q_coord_2x = F.grid_sample(
            feat2x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_4x = F.grid_sample(
            feat4x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
        q_coord_8x = F.grid_sample(
            feat8x_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
              
            
        # 将2x 4x 8x特征一起做k
        #K: b q c scales  -> 分多头 b q h c scales
        scales_k_feat = torch.stack([q_feat_2x, q_feat_4x, q_feat_8x], dim=-1)
        scales = scales_k_feat.shape[-1]
        
        scales_v_feat = scales_k_feat.clone().permute(0, 1, 3, 2) # V: b q c scales  ->  b q scales c 
        
        scales_k_feat = scales_k_feat.reshape(
            bs, q, self.head, self.dim // self.head, scales
        )
        
        # Q: b q 1 c -> 分多头 b q h 1 c
        q_feat = q_feat.unsqueeze(2).reshape(
            bs, q, 1, self.head, self.dim // self.head
        ).permute(0, 1, 3, 2, 4)
        
        scales_coord = torch.stack([q_coord_2x, q_coord_4x, q_coord_8x], dim=2) # 多个高分辨坐标
        
        # LR_coord在multi_coord上的偏移
        rel_coord = q_coord.unsqueeze(2) - scales_coord
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]
        
        # multi-scale cross attention
        _,pb = self.pe_encoder(rel_coord)
        
        attn = torch.matmul(q_feat, scales_k_feat).reshape(
            bs, q, self.head, scales
        ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)
        
        attn = F.softmax(torch.add(attn, pb), dim=-2) # 加入pos-bias
        
        attn = attn.reshape(bs, q, scales, self.head, 1)
        scales_v_feat = scales_v_feat.reshape(
            bs, q, scales, self.head, self.dim // self.head
        )
        scales_v_feat = torch.mul(scales_v_feat, attn).reshape(bs, q, -1)
    
        if self.cell_decode:
            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]
            inp = torch.cat([scales_v_feat, rel_cell], dim=-1)
        
        pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)

        # LR残差连接
        pred += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return pred

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-multi-fft-pe')
class LIIF_multi_fft_pe(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,pe_spec=None,head=8,
                 local_ensemble=True, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        
        imnet_in_dim = self.encoder.out_dim
        
        # LR->HR 
        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )

        if imnet_spec is not None:
            # imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim}).cuda()
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat_1 = self.upsample_x2(feat) # 96
        feat_2 = self.upsample_x4(feat_1) # 192
        
        # 不用fpn直接得到四个尺度的特征
        feat = feat
        feat2x = feat_1
        feat4x = feat_2
        
        feat_fft = torch.fft.fft2(feat,dim=(-2,-1))
        feat_fft2x = torch.fft.fft2(feat2x, dim=(-2, -1))
        feat_fft4x = torch.fft.fft2(feat4x, dim=(-2, -1))
        
        h1,w1 = feat_fft.shape[-2:]
        h2,w2 = feat_fft2x.shape[-2:]
        h3,w3 = feat_fft4x.shape[-2:]
        
        feat_fft_shift = torch.fft.fftshift(feat_fft, dim=(-2, -1))
        feat_fft2x_shift = torch.fft.fftshift(feat_fft2x, dim=(-2, -1))
        feat_fft4x_shift = torch.fft.fftshift(feat_fft4x, dim=(-2, -1))
        
        feat_fft2x_shift[:,:,((h2-h1) // 2): ((h2-h1) // 2) + h1 , ((w2-w1) // 2): ((w2-w1) // 2) + w1 ] = feat_fft_shift
        feat_fft4x_shift[:,:,((h3-h2) // 2): ((h3-h2) // 2) + h2 , ((w3-w2) // 2): ((w3-w2) // 2) + w2 ] = feat_fft2x_shift
        
        feat = torch.real(torch.fft.ifft2(feat_fft4x_shift)) # 转为实数feature
        
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
                    
                # ------------------------------计算rel——coord
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                # 位置编码
                pe, _ = self.pe_encoder(rel_coord)
                q_feat.mul_(pe)
                
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_cell], dim=-1)

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
        
        # 残差似乎很有用
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)



@register('liif-multi-fft-pe-high')
class LIIF_multi_fft_pe_high(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,pe_spec=None,head=8,
                 local_ensemble=True, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        
        imnet_in_dim = self.encoder.out_dim
        
        # LR->HR 
        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )

        if imnet_spec is not None:
            # imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim}).cuda()
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat_1 = self.upsample_x2(feat) # 96
        feat_2 = self.upsample_x4(feat_1) # 192
        
        # 不用fpn直接得到四个尺度的特征
        feat = feat
        feat2x = feat_1
        feat4x = feat_2
        
        feat_fft = torch.fft.fft2(feat,dim=(-2,-1))
        feat_fft2x = torch.fft.fft2(feat2x, dim=(-2, -1))
        feat_fft4x = torch.fft.fft2(feat4x, dim=(-2, -1))
        
        h1,w1 = feat_fft.shape[-2:]
        h2,w2 = feat_fft2x.shape[-2:]
        h3,w3 = feat_fft4x.shape[-2:]
        # h0,w0 = h1//2, w1//2
        
        feat_fft_shift = torch.fft.fftshift(feat_fft, dim=(-2, -1))
        feat_fft2x_shift = torch.fft.fftshift(feat_fft2x, dim=(-2, -1))
        feat_fft4x_shift = torch.fft.fftshift(feat_fft4x, dim=(-2, -1))
        
        h0, w0 = h1//2 , w1//2
        feat_fft_shift[:,:,((h1-h0) // 2): ((h1-h0) // 2) + h0 , ((w1-w0) // 2): ((w1-w0) // 2) + w0 ] = 0
        feat_fft2x_shift[:,:,((h2-h1) // 2): ((h2-h1) // 2) + h1 , ((w2-w1) // 2): ((w2-w1) // 2) + w1 ] = feat_fft_shift
        feat_fft4x_shift[:,:,((h3-h2) // 2): ((h3-h2) // 2) + h2 , ((w3-w2) // 2): ((w3-w2) // 2) + w2 ] = feat_fft2x_shift
        
        
        feat = torch.real(torch.fft.ifft2(feat_fft4x_shift)) # 转为实数feature
        
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
                    
                # ------------------------------计算rel——coord
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                # 位置编码
                pe, _ = self.pe_encoder(rel_coord)
                q_feat.mul_(pe)
                
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_cell], dim=-1)

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
        
        # 残差似乎很有用
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)


@register('liif-multi-fft')
class LIIF_multi_fft(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,pe_spec=None,head=8,
                 local_ensemble=True, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        
        self.encoder = models.make(encoder_spec)
        self.head = head    
        self.pe_encoder = models.make(pe_spec)
        
        imnet_in_dim = self.encoder.out_dim
        
        # LR->HR 
        self.upsample_x2 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.upsample_x4 = nn.Sequential(
            nn.Conv2d(imnet_in_dim, 4 * imnet_in_dim, 3, 1, 1),
            nn.PixelShuffle(2)
        )

        if imnet_spec is not None:
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim}).cuda()
        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        feat_1 = self.upsample_x2(feat) # 96
        feat_2 = self.upsample_x4(feat_1) # 192
        
        # 不用fpn直接得到四个尺度的特征
        feat = feat
        feat2x = feat_1
        feat4x = feat_2
        
        feat_fft = torch.fft.fft2(feat,dim=(-2,-1))
        feat_fft2x = torch.fft.fft2(feat2x, dim=(-2, -1))
        feat_fft4x = torch.fft.fft2(feat4x, dim=(-2, -1))
        
        h1,w1 = feat_fft.shape[-2:]
        h2,w2 = feat_fft2x.shape[-2:]
        h3,w3 = feat_fft4x.shape[-2:]
        
        feat_fft_shift = torch.fft.fftshift(feat_fft, dim=(-2, -1))
        feat_fft2x_shift = torch.fft.fftshift(feat_fft2x, dim=(-2, -1))
        feat_fft4x_shift = torch.fft.fftshift(feat_fft4x, dim=(-2, -1))
        
        feat_fft2x_shift[:,:,((h2-h1) // 2): ((h2-h1) // 2) + h1 , ((w2-w1) // 2): ((w2-w1) // 2) + w1 ] = feat_fft_shift
        feat_fft4x_shift[:,:,((h3-h2) // 2): ((h3-h2) // 2) + h2 , ((w3-w2) // 2): ((w3-w2) // 2) + w2 ] = feat_fft2x_shift
        
        feat = torch.real(torch.fft.ifft2(feat_fft4x_shift)) # 转为实数feature
        
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
                    
                # ------------------------------计算rel——coord
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
        
        # 残差似乎很有用
        ret += F.grid_sample(self.inp, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)

