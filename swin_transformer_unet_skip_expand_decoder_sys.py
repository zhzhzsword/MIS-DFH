import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class CNNBranch(nn.Module):
    """
    CNN分支用于与SwinUnet编码器并行提取特征
    """

    def __init__(self, in_chans=3, base_channels=64, depths=[2, 2, 2, 2]):
        super(CNNBranch, self).__init__()

        # 初始卷积层，与SwinUnet的patch embedding匹配
        self.initial_conv = nn.Conv2d(in_chans, base_channels, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # CNN阶段，匹配SwinUnet编码器的各个阶段
        self.stages = nn.ModuleList()

        # 第一阶段特征
        self.stages.append(self._make_stage(base_channels, base_channels, depths[0], stride=1))

        # 剩余阶段，带下采样
        for i in range(1, len(depths)):
            in_channels = base_channels * (2 ** (i - 1))
            out_channels = base_channels * (2 ** i)
            self.stages.append(self._make_stage(in_channels, out_channels, depths[i], stride=2))

    def _make_stage(self, in_channels, out_channels, blocks, stride):
        layers = []

        # 在除第一阶段外的每个阶段开始进行下采样
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            downsample = None

        # 添加第一个块，可能包含下采样
        layers.append(ResBlock(in_channels, out_channels, stride, downsample))

        # 添加剩余块
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 存储每个阶段的特征
        features = []

        # 初始卷积
        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 通过每个阶段处理并收集特征
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


class ResBlock(nn.Module):
    """
    基本的ResNet块
    """

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FeatureFusionModule(nn.Module):
    """
    特征融合模块，用于融合CNN分支和SwinUnet编码器的特征
    """

    def __init__(self, dims=[96, 192, 384, 768]):
        super(FeatureFusionModule, self).__init__()

        # CNN特征的精炼层
        self.refinement_layers = nn.ModuleList()
        for dim in dims:
            self.refinement_layers.append(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ))

        # 特征融合的注意力门
        self.attention_gates = nn.ModuleList()
        for dim in dims:
            self.attention_gates.append(AttentionGate(dim, dim))

    def forward(self, cnn_features, swin_features):
        """
        cnn_features: CNN分支特征列表
        swin_features: SwinUnet编码器特征列表（2D格式）
        """
        fused_features = []

        for i, (cnn_feat, refine_layer, attn_gate) in enumerate(
                zip(cnn_features, self.refinement_layers, self.attention_gates)):
            # 获取对应的Swin特征并在需要时重塑为2D格式
            swin_feat = swin_features[i]
            if len(swin_feat.shape) == 3:  # 如果形状为[B, L, C]
                B, L, C = swin_feat.shape
                H = W = int(L ** 0.5)
                swin_feat = swin_feat.permute(0, 2, 1).reshape(B, C, H, W)

            # 精炼CNN特征
            refined_cnn = refine_layer(cnn_feat)

            # 使用注意力门融合特征
            fused = attn_gate(refined_cnn, swin_feat)

            # 在需要时重塑回原始Swin特征格式
            if len(swin_features[i].shape) == 3:
                B, C, H, W = fused.shape
                fused = fused.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

            fused_features.append(fused)

        return fused_features


class AttentionGate(nn.Module):
    """
    注意力门，用于特征融合
    """

    def __init__(self, cnn_channels, swin_channels):
        super(AttentionGate, self).__init__()

        self.cnn_conv = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=1)
        self.swin_conv = nn.Conv2d(swin_channels, cnn_channels, kernel_size=1)
        self.fusion_conv = nn.Conv2d(cnn_channels, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cnn_feat, swin_feat):
        """
        用于融合CNN和Swin特征的注意力机制
        """
        # 转换特征
        cnn_transformed = self.cnn_conv(cnn_feat)
        swin_transformed = self.swin_conv(swin_feat)

        # 计算注意力图
        fusion = self.relu(cnn_transformed + swin_transformed)
        attention_map = self.sigmoid(self.fusion_conv(fusion))

        # 应用注意力并组合特征
        attended_cnn = cnn_feat * attention_map

        # 与Swin特征融合
        fused = attended_cnn + swin_feat

        return fused


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim) if LayerNorm_type == 'WithBias' else nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x):
        # 输入x的形状是[B, C, H, W]
        # 转换为[B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        # 应用LayerNorm
        x = self.norm(x)
        # 转回[B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x


class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()
        self.dim = dim  # 保存输入维度
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.patch_size = 8

    def forward(self, x):
        # 如果输入通道数与初始化时的通道数不匹配，动态调整卷积层
        if x.size(1) != self.dim:
            print(f"Recreating FSAS for channel dimension {x.size(1)} (was {self.dim})")
            self.dim = x.size(1)
            device = x.device

            # 重新创建卷积层以匹配输入通道数
            self.to_hidden = nn.Conv2d(self.dim, self.dim * 6, kernel_size=1, bias=False).to(device)
            self.to_hidden_dw = nn.Conv2d(self.dim * 6, self.dim * 6, kernel_size=3, stride=1, padding=1,
                                          groups=self.dim * 6, bias=False).to(device)
            self.project_out = nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False).to(device)
            self.norm = LayerNorm(self.dim * 2, LayerNorm_type='WithBias').to(device)

        # 正常的FSAS处理流程
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # 检查特征图尺寸并调整patch_size
        h, w = x.shape[2], x.shape[3]

        try:
            # 使用自适应的特征处理方法
            # 1. 使用1x1卷积进行特征变换
            q_trans = nn.Conv2d(q.size(1), q.size(1), kernel_size=1).to(x.device)(q)
            k_trans = nn.Conv2d(k.size(1), k.size(1), kernel_size=1).to(x.device)(k)

            # 2. 使用全局池化提取特征上下文
            q_pool = F.adaptive_avg_pool2d(q, (h // 2, w // 2))
            k_pool = F.adaptive_avg_pool2d(k, (h // 2, w // 2))

            # 3. 使用FFT进行频域交互
            q_fft = torch.fft.rfft2(q_pool.float())
            k_fft = torch.fft.rfft2(k_pool.float())

            # 4. 频域相乘
            out_fft = q_fft * k_fft

            # 5. 转回空间域
            out = torch.fft.irfft2(out_fft, s=(h // 2, w // 2))

            # 6. 上采样到原始尺寸
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

            # 7. 计算特征维度并确保out维度正确
            feature_dim = out.size(1)
            if feature_dim != self.dim * 2:
                # 使用1x1卷积调整通道数
                channel_adj = nn.Conv2d(feature_dim, self.dim * 2, kernel_size=1).to(x.device)
                out = channel_adj(out)

            # 8. 正则化和通道融合
            out = self.norm(out)
            output = v * out
            output = self.project_out(output)

            return output + x  # 残差连接

        except Exception as e:
            print(f"Error in FSAS processing: {e}")
            # 如果处理失败，返回原始输入
            return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print(x.shape)
        H, W = self.input_resolution
        x = self.expand(x)
        # print(x.shape)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # print(x.shape)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        # print(x.shape)
        x = x.view(B, -1, C // 4)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)

        return x


class my_PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        print(x.shape)
        x = self.expand(x)
        print(x.shape)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, flag=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.GJFH = GJFH()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        self.flag = flag
        self.convBottleneckBlock = ConvBottleneckBlock(768, 768, 768)

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            # else:
            #     x = blk(x)
            else:
                if not self.flag:
                    x = blk(x)
        if self.flag:
            # print(x.shape)
            x = x.reshape(-1, 7, 7, 768)  # iteration 7:([10, 768, 7, 7])
            # print(x.shape)
            x = x.permute(0, 3, 1, 2)
            # print(x.shape)
            x = self.GJFH(x)
            # print(x.shape)
        if self.downsample is not None:
            x = self.downsample(x)
            # print(x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ConvBottleneckBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 1, padding=0)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class SwinTransformerSysWithCNN(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", use_cnn_branch=True, **kwargs):
        super().__init__()

        # 初始化原始SwinUnet组件
        print(
            "SwinTransformerSysWithCNN expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths, depths_decoder, drop_path_rate, num_classes))
        # 添加ds_heads初始化
        self.ds_heads = None  # 初始为None，在第一次调用时创建
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patch_size = patch_size
        self.use_cnn_branch = use_cnn_branch

        # 将图像分割为非重叠的patch
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规则

        # 构建编码器和瓶颈层
        self.layers = nn.ModuleList()
        self.count = 1
        for i_layer in range(self.num_layers):
            if self.count != 4:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint, flag=False)
                self.count += 1
            elif self.count == 4:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint, flag=True)
            self.layers.append(layer)

        # 构建解码器层
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        # 添加CNN分支和特征融合组件
        if use_cnn_branch:
            dims = [embed_dim * (2 ** i) for i in range(self.num_layers)]
            self.cnn_branch = CNNBranch(in_chans=in_chans, base_channels=embed_dim, depths=depths)
            self.feature_fusion = FeatureFusionModule(dims=dims)

        # 初始化权重
        self.apply(self._init_weights)

        # SwinUnet的原始组件
        self.my_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        nb_filter = [96, 192, 384, 768]
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[1], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])

        # 跳跃连接的FSAS模块
        self.skip_fsas = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                skip_channel = int(embed_dim * 2 ** (self.num_layers - 1)) // 2
            else:
                skip_channel = int(embed_dim * 2 ** (self.num_layers - 1 - i))
            self.skip_fsas.append(FSAS(dim=skip_channel))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # 并行运行CNN分支与SwinUnet编码器
        cnn_features = self.cnn_branch(x) if self.use_cnn_branch else None

        # 原始SwinUnet编码器前向路径
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x_downsample = []
        swin_features = []

        # 通过编码器层处理并收集特征
        for i, layer in enumerate(self.layers):
            B, L, C = x.shape
            H, W = int(math.sqrt(L)), int(math.sqrt(L))

            # 保存特征用于跳跃连接
            x_downsample.append(x.view(B, H, W, C))

            # 跟踪Swin特征用于融合
            swin_feat = x.clone()
            swin_features.append(swin_feat)

            # 应用层
            x = layer(x)

        # 添加最终特征
        x_last = x.permute(0, 2, 3, 1) if len(x.shape) == 4 else x
        if len(x_last.shape) == 3:
            B, L, C = x_last.shape
            H = W = int(math.sqrt(L))
            x_last = x_last.view(B, H, W, C)
        x_downsample.append(x_last)

        if self.use_cnn_branch:
            # 融合CNN特征与Swin特征
            fused_features = self.feature_fusion(cnn_features, swin_features)

            # 用融合特征替换原始编码器特征
            for i in range(len(swin_features)):
                if isinstance(x_downsample[i], torch.Tensor):
                    # 转回网络其余部分期望的格式
                    if len(fused_features[i].shape) == 3:  # [B, L, C]
                        B, L, C = fused_features[i].shape
                        H = W = int(math.sqrt(L))
                        x_downsample[i] = fused_features[i].view(B, H, W, C)
                    else:  # [B, C, H, W]
                        B, C, H, W = fused_features[i].shape
                        x_downsample[i] = fused_features[i].permute(0, 2, 3, 1)

        # 处理x_downsample为原始SwinUnet连接
        B, H, W, C = x_last.shape
        x = x_last.reshape(B, H * W, C)
        x = self.norm(x)  # B L C

        # 原始SwinUnet连接
        x0_1 = self.conv0_1(
            torch.cat([x_downsample[0].permute(0, 3, 1, 2), self.my_up(x_downsample[1].permute(0, 3, 1, 2))], 1))
        x1_1 = self.conv1_1(
            torch.cat([x_downsample[1].permute(0, 3, 1, 2), self.my_up(x_downsample[2].permute(0, 3, 1, 2))], 1))
        x0_2 = self.conv0_2(torch.cat([x_downsample[0].permute(0, 3, 1, 2), x0_1, self.my_up(x1_1)], 1))

        # 确保索引在范围内
        idx = min(4, len(x_downsample) - 1)
        x2_1 = self.conv2_1(
            torch.cat([x_downsample[2].permute(0, 3, 1, 2), self.my_up(x_downsample[idx].permute(0, 3, 1, 2))], 1))

        x1_2 = self.conv1_2(torch.cat([x_downsample[1].permute(0, 3, 1, 2), x1_1, self.my_up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x_downsample[0].permute(0, 3, 1, 2), x0_1, x0_2, self.my_up(x1_2)], 1))

        x_downsample_new = []
        x_downsample_new.append(torch.flatten(x0_3, start_dim=2, end_dim=-1).permute(0, 2, 1))
        x_downsample_new.append(torch.flatten(x1_2, start_dim=2, end_dim=-1).permute(0, 2, 1))
        x_downsample_new.append(torch.flatten(x2_1, start_dim=2, end_dim=-1).permute(0, 2, 1))

        return x, x_downsample_new

    def forward_up_features(self, x, x_downsample):
        # 原始SwinTransformerSys中的代码
        decoder_features = []

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # 获取跳跃连接特征
                skip_feature = x_downsample[3 - inx]

                # 使用FSAS处理（如果需要）
                B, L, C = skip_feature.shape
                H = W = int(math.sqrt(L))
                skip_feat_4d = skip_feature.reshape(B, H, W, C).permute(0, 3, 1, 2)

                try:
                    skip_feat_4d = self.skip_fsas[inx - 1](skip_feat_4d)
                except Exception as e:
                    print(f"FSAS处理失败: {e}")

                skip_feature = skip_feat_4d.permute(0, 2, 3, 1).reshape(B, L, C)

                # 连接并处理
                x = torch.cat([x, skip_feature], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

            decoder_features.append(x)

        x = self.norm_up(x)
        return x, decoder_features

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x, deep_supervision=True):
        # 通过带有CNN分支集成的编码器前向传播
        x, x_downsample = self.forward_features(x)

        # 通过解码器前向传播
        x, decoder_features = self.forward_up_features(x, x_downsample)

        # 最终上采样和输出
        output = self.up_x4(x)

        # 始终进行深度监督计算，返回主输出和深度监督输出的列表
        deep_outputs = self.get_deep_supervision_outputs(decoder_features)

        # 返回包含主输出和深度监督输出的列表
        return [output] + deep_outputs
    # 初始化深度监督头（延迟初始化）
    def init_deep_supervision_heads(self, decoder_features):
        if self.ds_heads is None:
            # print("初始化深度监督头...")
            self.ds_heads = nn.ModuleList()

            for i, feat in enumerate(decoder_features):
                B, L, C = feat.shape
                # print(f"解码器层 {i} 实际通道数: {C}")
                # 为每个解码器层创建对应通道数的深度监督头
                ds_head = nn.Conv2d(C, self.num_classes, kernel_size=1, bias=False)
                # 将头移动到与模型相同的设备上
                ds_head = ds_head.to(feat.device)
                self.ds_heads.append(ds_head)

            # print("深度监督头初始化完成")

    # 将中间特征转换为与输出相同大小的预测图
    def get_deep_supervision_outputs(self, decoder_features):
        # 首次运行时初始化深度监督头
        if self.ds_heads is None:
            self.ds_heads = nn.ModuleList()
            for i, feat in enumerate(decoder_features):
                B, L, C = feat.shape
                # 为每个解码器层创建对应通道数的深度监督头
                ds_head = nn.Conv2d(C, self.num_classes, kernel_size=1, bias=False)
                # 将头移动到与模型相同的设备上
                ds_head = ds_head.to(feat.device)
                self.ds_heads.append(ds_head)

        # 处理每个解码器特征
        deep_outputs = []
        final_size = (self.patches_resolution[0] * 4, self.patches_resolution[1] * 4)

        for i, feat in enumerate(decoder_features):
            # 将特征从序列形式转换为空间形式
            B, L, C = feat.shape
            H = W = int(math.sqrt(L))
            feat_map = feat.reshape(B, H, W, C).permute(0, 3, 1, 2)

            # 应用深度监督头
            feat_out = self.ds_heads[i](feat_map)

            # 上采样到与最终输出相同的尺寸
            if feat_out.shape[2:] != final_size:
                feat_out = nn.functional.interpolate(
                    feat_out,
                    size=final_size,
                    mode='bilinear',
                    align_corners=True
                )

            deep_outputs.append(feat_out)

        return deep_outputs
class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        self.count = 1
        for i_layer in range(self.num_layers):
            if self.count != 4:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint, flag=False)
                self.count += 1
            elif self.count == 4:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   depth=depths[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint, flag=True)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                             patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                             patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        # 添加深度监督头
        # 首先在前向传播中收集实际的解码器层通道数，然后再初始化深度监督头
        self.ds_heads = None  # 先设为None，在第一次前向传播后初始化

        self.apply(self._init_weights)

        # 原有的额外组件
        self.my_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        nb_filter = [96, 192, 384, 768]
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[1], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])

        # 添加FSAS模块到跳跃连接
        self.skip_fsas = nn.ModuleList()
        # 为每个解码器层创建一个FSAS模块
        for i in range(self.num_layers):
            # 计算每个跳跃连接层的通道数
            if i == 0:
                skip_channel = int(embed_dim * 2 ** (self.num_layers - 1)) // 2
            else:
                skip_channel = int(embed_dim * 2 ** (self.num_layers - 1 - i))

            self.skip_fsas.append(FSAS(dim=skip_channel))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            B, L, C = x.shape
            H, W = int(math.sqrt(L)), int(math.sqrt(L))

            x_downsample.append(x.view(B, H, W, C))
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x_downsample.append(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = self.norm(x)  # B L C

        x0_1 = self.conv0_1(
            torch.cat([x_downsample[0].permute(0, 3, 1, 2), self.my_up(x_downsample[1].permute(0, 3, 1, 2))], 1))
        x1_1 = self.conv1_1(
            torch.cat([x_downsample[1].permute(0, 3, 1, 2), self.my_up(x_downsample[2].permute(0, 3, 1, 2))], 1))
        x0_2 = self.conv0_2(torch.cat([x_downsample[0].permute(0, 3, 1, 2), x0_1, self.my_up(x1_1)], 1))

        # 修复索引，确保正确访问
        idx = min(4, len(x_downsample) - 1)  # 防止索引超出范围
        x2_1 = self.conv2_1(
            torch.cat([x_downsample[2].permute(0, 3, 1, 2), self.my_up(x_downsample[idx].permute(0, 3, 1, 2))], 1))

        x1_2 = self.conv1_2(torch.cat([x_downsample[1].permute(0, 3, 1, 2), x1_1, self.my_up(x2_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x_downsample[0].permute(0, 3, 1, 2), x0_1, x0_2, self.my_up(x1_2)], 1))

        x_downsample_new = []
        x_downsample_new.append(torch.flatten(x0_3, start_dim=2, end_dim=-1).permute(0, 2, 1))
        x_downsample_new.append(torch.flatten(x1_2, start_dim=2, end_dim=-1).permute(0, 2, 1))
        x_downsample_new.append(torch.flatten(x2_1, start_dim=2, end_dim=-1).permute(0, 2, 1))

        return x, x_downsample_new

    # 解码器部分，收集中间特征用于深度监督
    def forward_up_features(self, x, x_downsample):
        # 存储解码器的中间特征，用于深度监督
        decoder_features = []

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # 获取跳跃连接特征
                skip_feature = x_downsample[3 - inx]

                # 将跳跃连接特征转换为适合FSAS处理的形状
                B, L, C = skip_feature.shape
                H = W = int(math.sqrt(L))
                skip_feat_4d = skip_feature.reshape(B, H, W, C).permute(0, 3, 1, 2)

                # 应用FSAS处理
                try:
                    skip_feat_4d = self.skip_fsas[inx - 1](skip_feat_4d)
                except Exception as e:
                    print(f"FSAS处理失败: {e}")
                    # 如果处理失败，保持原样

                # 转回原始形状
                skip_feature = skip_feat_4d.permute(0, 2, 3, 1).reshape(B, L, C)

                # 连接当前特征和处理后的跳跃连接特征
                x = torch.cat([x, skip_feature], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

            # 存储解码器每一层的特征用于深度监督
            decoder_features.append(x)

        x = self.norm_up(x)  # B L C

        return x, decoder_features

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    # 初始化深度监督头（延迟初始化）
    def init_deep_supervision_heads(self, decoder_features):
        if self.ds_heads is None:
            # print("初始化深度监督头...")
            self.ds_heads = nn.ModuleList()

            for i, feat in enumerate(decoder_features):
                B, L, C = feat.shape
                # print(f"解码器层 {i} 实际通道数: {C}")
                # 为每个解码器层创建对应通道数的深度监督头
                ds_head = nn.Conv2d(C, self.num_classes, kernel_size=1, bias=False)
                # 将头移动到与模型相同的设备上
                ds_head = ds_head.to(feat.device)
                self.ds_heads.append(ds_head)

            # print("深度监督头初始化完成")

    # 将中间特征转换为与输出相同大小的预测图
    def get_deep_supervision_outputs(self, decoder_features):
        # 首次运行时初始化深度监督头
        if self.ds_heads is None:
            self.init_deep_supervision_heads(decoder_features)

        deep_outputs = []
        final_size = (self.patches_resolution[0] * 4, self.patches_resolution[1] * 4)

        for i, feat in enumerate(decoder_features):
            # 将特征从序列形式 (B, L, C) 转换为空间形式 (B, C, H, W)
            B, L, C = feat.shape
            H = W = int(math.sqrt(L))

            # 打印特征形状信息进行调试
            # print(f"解码器层 {i} 特征形状: ({B}, {L}, {C}), 转换为: ({B}, {C}, {H}, {W})")

            # 确保我们有正确的空间维度
            feat_map = feat.reshape(B, H, W, C).permute(0, 3, 1, 2)

            # 应用深度监督头
            feat_out = self.ds_heads[i](feat_map)

            # 上采样到与最终输出相同的尺寸
            if feat_out.shape[2:] != final_size:
                feat_out = nn.functional.interpolate(
                    feat_out,
                    size=final_size,
                    mode='bilinear',
                    align_corners=True
                )

            deep_outputs.append(feat_out)
            # print(f"输出形状{i + 1}: {feat_out.shape}")

        return deep_outputs

    # 前向传播以支持深度监督
    def forward(self, x, deep_supervision=True):
        # 编码器前向传播
        x, x_downsample = self.forward_features(x)

        # 解码器前向传播，收集中间特征
        x, decoder_features = self.forward_up_features(x, x_downsample)

        # 最终上采样和输出
        output = self.up_x4(x)

        # 如果需要深度监督，则返回所有中间层的预测
        if deep_supervision:
            deep_outputs = self.get_deep_supervision_outputs(decoder_features)
            return [output] + deep_outputs  # 返回列表：[最终输出, 深度监督输出1, 深度监督输出2, ...]
        else:
            return output  # 仅返回最终输出

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


#--------------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result



class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()
        # 保存 group_num 为实例属性
        self.group_num = group_num
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()
        # 保存 eps 为实例属性
        self.eps = 1e-10

    def forward(self, x):
        N, C, H, W = x.size()
        # 使用 reshape 代替 view
        x = x.reshape(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gn.weight + self.gn.bias

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)
    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
import torch.nn as nn
import torch
from einops import rearrange
import math

__all__ = ['AKConv', 'C2f_AKConv']

class AKConv(nn.Module):
    def __init__(self, inc, outc, num_param=1, stride=1, bias=None):
        super(AKConv, self).__init__()

        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU())  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the AKConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int), indexing='xy')
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number), indexing='xy')

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride), indexing='xy')

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)

        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # 根据实际情况调整
        index = index.clamp(min=0, max=x.shape[-1] - 1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = AKConv(c_, c2, k[1], 1, g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_AKConv(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x = self.cv1(x)
        x = x.chunk(2, 1)
        y = list(x)
        # y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SEMAConv(nn.Module):
    def __init__(self, channels=32 ):
        super(SEMAConv, self).__init__()
        self.ScConv = ScConv(op_channel=channels)
        self.EMA = EMA(channels=channels)
    def forward(self,x):
        x1 = self.ScConv(x)
        x2 = self.EMA(x)
        x = x1 + x2
        return x

class GJFH(nn.Module):
    def __init__(self, in_channels=768, out_channels=768):  # 修改通道数
        super(GJFH, self).__init__()
        self.ScConv = ScConv(op_channel=in_channels)  # 输入通道改为768
        self.AKconv = AKConv(inc=in_channels, outc=out_channels)  # 输入输出通道改为768
        self.SEMAConv = SEMAConv(channels=in_channels)  # 输入通道改为768
        self.CBAM = CBAM(in_planes=out_channels)  # 输入通道改为768
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.ScConv(x)  # C:768
        x2 = self.AKconv(x)  # C:768
        a = self.sigmod(x1 + x2)
        x3 = x1 * a + x2 * (1 - a)
        x3 = self.SEMAConv(x3)
        out = self.CBAM(x3) + x  # 残差连接保持通道一致
        return out

# if __name__ == '__main__':
#     input = torch.randn(24, 768, 7, 7)  # 修改输入形状
#     GJFH_model = GJFH(768, 768)  # 初始化参数改为768
#     output = GJFH_model(input)
#     print('input:', input.shape)
#     print('output:', output.shape)  # 应输出 torch.Size([24, 768, 7, 7])