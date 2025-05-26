import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=16, w=9):  # 调整h和w以适应256x256输入
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=16, w=9):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class Block_attention(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=16, w=9):
        super().__init__()
        num_heads = 6  # 4 for tiny, 6 for small and 12 for base
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_scale=False, attn_drop=drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding - 调整为256x256输入
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # 允许不同尺寸的输入，移除严格尺寸检查
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class UpSampleLayer(nn.Module):
    """ 上采样层，用于恢复空间尺寸
    """

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, x, skip_connection=None):
        # 上采样
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        # 如果有跳跃连接，进行融合
        if skip_connection is not None:
            x = x + skip_connection  # 或使用concat，根据需要调整

        x = self.conv(x)
        x = F.relu(x)
        return x


class SpectFormer(nn.Module):

    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=31, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of output channels for dense prediction
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            drop_rate (float): dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # 调整为256x256输入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size  # 256//16=16
        w = h // 2 + 1  # 16//2+1=9

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # 保存中间特征用于解码器
        self.encoder_features = []

        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i < 4:  # 前4层使用光谱门控块
                layer = Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                              norm_layer=norm_layer, h=h, w=w)
                self.blocks.append(layer)
                # 保存前几层特征用于上采样
                if i == 1 or i == 3:
                    self.encoder_features.append(None)
            else:  # 后面使用注意力块
                layer = Block_attention(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                                        norm_layer=norm_layer, h=h, w=w)
                self.blocks.append(layer)

        self.norm = norm_layer(embed_dim)

        # 解码器部分 - 用于恢复空间尺寸
        self.decoder = nn.Sequential(
            UpSampleLayer(embed_dim, embed_dim // 2, scale_factor=2),
            UpSampleLayer(embed_dim // 2, embed_dim // 4, scale_factor=2),
            UpSampleLayer(embed_dim // 4, embed_dim // 8, scale_factor=2),
            nn.Conv2d(embed_dim // 8, num_classes, kernel_size=1)
        )

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 保存中间特征用于解码器
        skip_connections = []

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # 保存第2和第4层的特征
            if i == 1 or i == 3:
                # 转换为2D特征图用于跳跃连接
                spatial_size = int(math.sqrt(x.size(1)))
                feature_map = x.view(B, spatial_size, spatial_size, -1).permute(0, 3, 1, 2)
                skip_connections.append(feature_map)

        # 将最后一层特征转换为2D特征图
        spatial_size = int(math.sqrt(x.size(1)))
        x = x.view(B, spatial_size, spatial_size, -1).permute(0, 3, 1, 2)

        return x, skip_connections

    def forward(self, x):
        # 提取特征
        x, skip_connections = self.forward_features(x)

        # 解码器上采样过程
        x = self.decoder[0](x, skip_connections[1] if len(skip_connections) > 1 else None)
        x = self.decoder[1](x, skip_connections[0] if len(skip_connections) > 0 else None)
        x = self.decoder[2](x)
        x = self.decoder[3](x)

        # 确保输出尺寸与输入匹配
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x
