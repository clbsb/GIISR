import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from fairscale.nn import checkpoint_wrapper

from basicsr.utils.registry import ARCH_REGISTRY

# ================
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

# ================================CA=======================================================

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CA(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CA, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):

        return self.cab(x) + x



# ================================CategoryAttention=======================================================
class FeatureRefine(nn.Module):
    def __init__(self, dim, cnn_size=13):
        super(FeatureRefine, self).__init__()

        # 定义卷积层
        self.fr = nn.Sequential(nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=cnn_size, groups=dim, padding=(cnn_size-1)//2),
                                  nn.GELU(),
                                  nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1),
                                  nn.GELU())

        self.dsfr = nn.Sequential(nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=cnn_size, groups=dim, padding=(cnn_size - 1) // 2),)

        # self.dsfr = nn.Sequential(nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=cnn_size, groups=dim, padding=(cnn_size-1)//2),
        #                         nn.GELU(),
        #                         nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1))

    def tokensort(self, x, cate, c):
        _, indices = torch.sort(cate, dim=1, descending=True)
        x_sorted = torch.gather(x, 1, indices.expand(-1, -1, c))
        return indices, x_sorted

    def tokenresort(self, x, indices, c):
        _, reverse_indices = torch.sort(indices, dim=1)
        x_restored = torch.gather(x, 1, reverse_indices.expand(-1, -1, c))
        return  x_restored

    def forward(self, x, cscore):
        b, n, c = x.size()
        # 依据得分排序，获得索引
        indices, x_sorted = self.tokensort(x, cscore, c)

        # 进行特征细化
        x_sorted = x_sorted.transpose(1, 2).contiguous()
        x_fr = self.fr(x_sorted)
        x_ds = F.interpolate(x_fr, size=512, mode='linear', align_corners=True)
        x_ds = self.dsfr(x_ds)
        x_ds = F.interpolate(x_ds, size=n, mode='linear', align_corners=True)
        cnn_result = x_fr + x_ds
        cnn_result = cnn_result.transpose(1, 2).contiguous()

        # 根据索引恢复排序
        x_fr = self.tokenresort(cnn_result, indices, c)

        return x_fr

class EGFR(nn.Module):
    def __init__(self, dim, cnn_size=13): #  分类得分
        super(EGFR, self).__init__()

        # 定义分类打分线性层
        self.linear_pre = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU())
        self.linear_a = nn.Linear(dim//2, 1)
        self.linear_b = nn.Linear(dim//2, 1)
        self.linear_c = nn.Linear(dim//2, 1)
        self.softmax = nn.Softmax(dim=1)

        # 特征细化
        self.fr = FeatureRefine(dim, cnn_size)

        # 专家权重
        self.gp = nn.AdaptiveAvgPool1d(1)
        self.expert_weight = nn.Linear(dim, 3, bias=False)


    def forward(self, x): # (b,n,c)
        b, n, c = x.size()

        # 得出专家权重
        x_w = x.transpose(1, 2).contiguous() # (b,c,n)
        x_w = self.gp(x_w).squeeze(-1) # (b,c)
        x_w = self.expert_weight(x_w) # (b,3)
        x_w = F.softmax(x_w, dim=1, dtype=torch.float).to(x.dtype) # (b,3)
        x_w = x_w.unsqueeze(2) # (b,3,1)
        x_w = x_w.expand(-1, -1, c) # (b,3,c)

        # 预得分
        score_pre = self.linear_pre(x)

        # 专家打分排序引导特征细化
        score_a = self.linear_a(score_pre)
        x_fr_a = self.fr(x, score_a)
        x_fr_a_fin = x_fr_a + self.softmax(score_a).expand(-1, -1, c) * x

        score_b = self.linear_b(score_pre)
        x_fr_b = self.fr(x, score_b)
        x_fr_b_fin = x_fr_b + self.softmax(score_b).expand(-1, -1, c) * x

        score_c = self.linear_c(score_pre)
        x_fr_c = self.fr(x, score_c)
        x_fr_c_fin = x_fr_c + self.softmax(score_c).expand(-1, -1, c) * x

        x_fin = torch.stack([x_fr_a_fin, x_fr_b_fin, x_fr_c_fin], dim=1) # (b,3,n,c)
        x_fin = (x_fin * x_w.unsqueeze(2)).sum(dim=1) # (b,n,c)

        # torch.save(score_a, 'cate_a.pth')
        # torch.save(score_b, 'cate_b.pth')
        # torch.save(score_c, 'cate_c.pth')

        return x_fin


# ================================Upsample=======================================================

class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

# ================================layer=======================================================

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 ):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c).contiguous()
        x = self.proj(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 mlp_ratio,
                 norm_layer,
                 shift_size,
                 convffn_kernel_size,
                 act_layer=nn.GELU,
                 ):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size

        self.attn = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,
                               act_layer=act_layer)

        self.cnncate = EGFR(dim=dim, cnn_size=13)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size, params):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x) # (b,n,c)

        # 分类卷积
        x_cnncate = self.cnncate(x)

        x = x.view(b, h, w, c)

        # 滑动
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 切割为窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # 计算窗口自注意力
        if self.shift_size > 0:
            attn_windows = self.attn(x_windows, rpi=params['rpi_sa'], mask=params['attn_mask'])
        else:
            attn_windows = self.attn(x_windows, rpi=params['rpi_sa'], mask=None)

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # 滑动还原
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(b, h * w, c) + shortcut + x_cnncate

        # FFN
        x = x + self.convffn(self.norm2(x), x_size)

        return x

# ================================block=======================================================
class BasicBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_layer,
                 num_heads,
                 window_size,
                 mlp_ratio,
                 norm_layer,
                 convffn_kernel_size,):
        super().__init__()
        self.dim = dim

        # if resi_connection == '1conv':
        #     self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # elif resi_connection == '3conv':
        #     # to save parameters and memory
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.conv = CA(num_feat=dim, compress_ratio=3, squeeze_factor=10) #squeeze_factor=30

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            layer = BasicLayer(dim=dim,
                               num_heads=num_heads,
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               norm_layer=norm_layer,
                               shift_size=0 if (i % 2 == 0) else window_size // 2,
                               convffn_kernel_size=convffn_kernel_size,)
            self.layers.append(layer)


    def forward(self, x, x_size, params):
        ori_x = x

        # embed
        x = x.flatten(2).transpose(1, 2).contiguous()

        for layer in self.layers:
            x = layer(x, x_size, params)

        # unembed
        x = x.transpose(1, 2).view(x.shape[0], self.dim, x_size[0], x_size[1]).contiguous()

        x = self.conv(x)
        x = x + ori_x

        return x

# ================================main=======================================================

@ARCH_REGISTRY.register()
class ESGFR_CA(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=60,
                 num_basicblock = 4,
                 num_layer=6,
                 num_heads=6,
                 window_size=8,
                 mlp_ratio=1.,
                 norm_layer=nn.LayerNorm,
                 upscale=2,
                 img_range=1.,
                 resi_connection='1conv',
                 convffn_kernel_size=7,):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.img_range = img_range
        self.window_size = window_size
        self.dim = embed_dim
        self.upscale = upscale

        # 预处理
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # build Residual Adaptive Token Dictionary Blocks (ATDB)
        self.blocks = nn.ModuleList()
        for i in range(num_basicblock):
            block = BasicBlock(dim=embed_dim,
                               num_layer=num_layer,
                               num_heads=num_heads,
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               norm_layer=norm_layer,
                               convffn_kernel_size=convffn_kernel_size,)
            self.blocks.append(block)
        self.norm = norm_layer(embed_dim)

        self.after_blocks = nn.Sequential(nn.Conv2d(embed_dim * num_basicblock, embed_dim, 1), nn.LeakyReLU(inplace=True))

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        self.apply(self._init_weights)

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])

        dense = []
        for block in self.blocks:
            x = block(x, x_size, params)
            dense.append(x)

        x = self.after_blocks(torch.cat(dense, dim=1))

        # 为了norm进行embed和unembed
        x = x.flatten(2).transpose(1, 2)
        x = x.contiguous()
        x = self.norm(x)  # b seq_len c
        x = x.transpose(1, 2).view(x.shape[0], self.dim, x_size[0], x_size[1]).contiguous()

        return x

    def forward(self, x):
        # padding
        H, W = x.shape[2:]
        res = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        x = self.check_image_size(x)

        _,_, h, w = x.size()

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x, params)) + x
        x = self.upsample(x)
        x = x[:, :, :H * self.upscale, :W * self.upscale] + res
        x = x / self.img_range + self.mean

        # unpadding
        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    # 初始化网络参数（权重和偏置）
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask



def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows



# ===================================================
def save_feature_maps(x, output_dir):
    """
    将输入特征图的每一个通道保存为灰度图。

    参数:
    x: torch.Tensor - 大小为 (b, c, h, w) 的特征图
    output_dir: str - 用于保存灰度图的目标文件夹
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 将特征图从 GPU 转移到 CPU 并转换为 NumPy 数组
    x = x.detach().cpu().numpy()

    batch_size, channels, height, width = x.shape

    for b in range(batch_size):
        for c in range(channels):
            # 获取单个通道的特征图
            feature_map = x[b, c, :, :]

            # 归一化处理
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            feature_map = (feature_map * 255).astype(np.uint8)

            # 构造保存路径
            filename = os.path.join(output_dir, f'batch_{b}_channel_{c}.png')

            # 使用 PIL 保存灰度图
            image = Image.fromarray(feature_map)
            image.save(filename)


