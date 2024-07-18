import torch
import torch.nn as nn
from math import ceil
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath

def weights_init_swin(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        trunc_normal_(m.weight.data, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("Conv") != -1:
        trunc_normal_(m.weight.data, mean=0.0, std=.02)
    elif classname.find("GroupNorm") != -1:
        trunc_normal_(m.weight.data, mean=1.0, std=.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("LayerNorm") != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)

def window_partition(patches, window_size):
    B, D, H, W, C = patches.shape
    x = patches.contiguous().view(B, C, D // window_size, window_size, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # (B, D//W, H//W, W//W, W, W, W, C)
    x = x.contiguous().view(-1, window_size**3, C)  # (num_windows*B, W*W*W, C)
    return x

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.contiguous().view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x
    
class CNNDouble(nn.Module):
    def __init__(self, scale_factor, in_size, out_size, dropout=0.0):
        super(CNNDouble, self).__init__()
        
        layers = [
            nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True),
            nn.Conv3d(in_size, out_size, 3, stride=1, padding=1),
            nn.GroupNorm(48, out_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_size, out_size, 3, 1, 1),
            nn.GroupNorm(48, out_size),
            nn.Dropout(dropout)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = self.model(x)
        return rearrange(x, 'b c d h w -> b d h w c')

class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).contiguous().view(-1, self.channels, size, size, size)

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.contiguous().view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.contiguous().view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.norm = nn.LayerNorm(in_channels * downscaling_factor ** 3)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 3, out_channels)

    def forward(self, x):
        b, _, h, _, _= x.shape
        size = h // self.downscaling_factor
        x = x.unfold(1, self.downscaling_factor, self.downscaling_factor).unfold(2,self.downscaling_factor, self.downscaling_factor).unfold(3,self.downscaling_factor, self.downscaling_factor)
        x = x.contiguous().view(b, size, size, size, -1)
        x = self.norm(x)
        x = self.linear(x)
        return x

class PatchExpand(nn.Module):
    def __init__(self, in_channels, scale, out_channels, in_dimension):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        #self.up_conv = nn.Conv1d(in_dimension**3, (in_dimension*scale)**3, 1, 1, 0)
        self.up_linear = nn.Linear(in_dimension**3, (in_dimension*scale)**3)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        B, D, _, _, C = x.shape
        x = x.contiguous().view(B,C,-1)
        x = self.up_linear(x) # B, C, D*D*D
        D = ceil(x.shape[2]**(1/3))
        
        x = x.permute(0,2,1) # B, D*D*D, C
        x = self.norm(x)
        x = self.linear(x)
        x = x.view(B,D,D,D,-1)
        return x

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement, self.displacement), dims=(1, 2, 3))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class PatchEmbed3D(nn.Module):

    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=96):
        super().__init__()
        
        self.in_chans=in_chans
        self.embed_dim=embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b c d h w -> b d h w c')
    
class PixelShuffle3d(nn.Module):
    def __init__(self, scale):
        super(PixelShuffle3d, self).__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_height, in_width, in_depth = input.size()            
        nOut = channels // (self.scale ** 3)
        out_height = in_height * self.scale
        out_width  = in_width  * self.scale
        out_depth  = in_depth  * self.scale
        
        input_view = input.view(batch_size, nOut, self.scale, self.scale, self.scale, in_height, in_width, in_depth)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_height, out_width, out_depth)
    
class UpsampleOneStep(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution, dropout=0.1):
        super(UpsampleOneStep,self).__init__()
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        
        m = [
            nn.Conv3d(num_feat, (scale ** 3) * num_feat, 1, 1, 0),
            nn.GELU(),
            PixelShuffle3d(scale),
            nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        ]
        if dropout: m.append(nn.Dropout(dropout))
        if num_out_ch==1: m.append(nn.Sigmoid())

        self.model = nn.Sequential(*m)
        
    def forward(self, x):
        x = x.permute(0,4,1,2,3).contiguous()
        return self.model(x).permute(0,2,3,4,1)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, window_size, dpr, dropout_attn, dropout_mlp):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention_block_1 = WindowAttention3D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            attn_drop=dropout_attn,
            proj_drop=dropout_attn
        )
        self.attention_block_2 = WindowAttention3D(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            attn_drop=dropout_attn,
            proj_drop=dropout_attn
        )
        self.mlp_block_1 = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout_mlp)))
        self.mlp_block_2 = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout_mlp)))
        self.forward_shift = CyclicShift(-window_size[0]//2)
        self.reverse_shift = CyclicShift(window_size[0]//2)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

    def forward(self, x, mask):

        B, D, H, W, _ = x.shape

        x = window_partition(x, self.window_size[0]) # (num_windows*B, W*W*W, C)

        # First attention : not shifted
        residual = x
        x = self.norm1(x)
        x = residual + self.drop_path(self.attention_block_1(x, mask=None))

        # First mlp
        x = self.drop_path(self.mlp_block_1(x))

        residual = x
        x = window_reverse(x, self.window_size, B, D, H, W)

        # reverse

        x = self.forward_shift(x)
        x = window_partition(x, self.window_size[0])
        x = self.norm2(x)
        x = residual + self.drop_path(self.attention_block_2(x, mask=mask))

        # Second mlp
        x = self.drop_path(self.mlp_block_2(x))

        x = window_reverse(x, self.window_size, B, D, H, W)

        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, downscaling_factor, num_heads, window_size, device, first_block=False, upsample=False, in_dimension=None, dropout_attn=0.1, dropout_mlp=0.5, dpr=0.1, mid=False, depth=1):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.device = device
        self.upsample = upsample
        self.flag = first_block
        self.mid = mid
        
        self.model = nn.ModuleList([])
        
        for i in range(depth):
            self.model.append(SwinBlock(dim=hidden_dimension, num_heads=num_heads,
                        mlp_dim=hidden_dimension * 4,
                        window_size=window_size,
                        dpr=dpr,
                        dropout_attn=dropout_attn,
                        dropout_mlp=dropout_mlp))

        if not upsample and not first_block and not mid:
                self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                                downscaling_factor=downscaling_factor)          
        elif upsample:
            self.patch_upsample = PatchExpand(in_channels=in_channels, scale=downscaling_factor, out_channels=hidden_dimension, in_dimension=in_dimension)        

    def compute_mask(self, D, H, W, window_size, device):
        shift_size = window_size[0]//2
        img_mask = torch.zeros((1, D, H, W, 1)).half().to(device)  # 1 Dp Hp Wp 1
        cnt = 0
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size), slice(-shift_size,None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size), slice(-shift_size,None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size), slice(-shift_size,None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, window_size[0])  # nW, ws[0]*ws[1]*ws[2], 1
        mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        
        # input : B, C, D, H, W

        if not self.upsample and not self.flag and not self.mid:
            x = self.patch_partition(x) # doubles the channel dim, halves the spatial dim
        
        if self.upsample:
            x = self.patch_upsample(x) # doubles the spatial dim, halves the channel dim
        
        # B, D, H, W, C
        _, D, H, W, _ = x.shape
        mask = self.compute_mask(D, H, W, self.window_size, device = self.device)

        for layer in self.model:
            x = layer(x, mask)
     
        return x


class SwinUNet(nn.Module):
    def __init__(self, device='cpu', in_dim=96, dropout_attn=0.1, dropout_mlp=0.1):
        super().__init__()

        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        self.sa = SelfAttention(1, 1)

        self.linear = nn.Sequential(
            nn.Linear(343*2, 343),
            nn.GELU()
        )
        
        self.patchemb1 = PatchEmbed3D(patch_size=(4,4,4), in_chans=1, embed_dim=in_dim)
        self.patchemb2 = PatchEmbed3D(patch_size=(4,4,4), in_chans=1, embed_dim=in_dim)

        self.stage1 = StageModule(in_channels=1, hidden_dimension=in_dim,
                                  downscaling_factor=2, num_heads=3, window_size=(7,7,7), device=device,
                                  dropout_attn=0.0, dropout_mlp=0.0, first_block=True, depth=2)
        self.stage2 = StageModule(in_channels=in_dim, hidden_dimension=in_dim*2,
                                  downscaling_factor=2, num_heads=6, window_size=(7,7,7), device=device,
                                  dropout_attn=dropout_attn, dropout_mlp=dropout_mlp, depth=2)
        self.stage3 = StageModule(in_channels=in_dim*2, hidden_dimension=in_dim*4,
                                  downscaling_factor=2, num_heads=12, window_size=(7,7,7), device=device,
                                  dropout_attn=dropout_attn, dropout_mlp=dropout_mlp, depth=2)

        self.Sstage1 = StageModule(in_channels=1, hidden_dimension=in_dim,
                                  downscaling_factor=2, num_heads=3, window_size=(7,7,7), device=device,
                                  dropout_attn=0.0, dropout_mlp=0.0, first_block=True, depth=2)
        self.Sstage2 = StageModule(in_channels=in_dim, hidden_dimension=in_dim*2,
                                  downscaling_factor=2, num_heads=6, window_size=(7,7,7), device=device,
                                  dropout_attn=dropout_attn, dropout_mlp=dropout_mlp, depth=2)
        self.Sstage3 = StageModule(in_channels=in_dim*2, hidden_dimension=in_dim*4,
                                  downscaling_factor=2, num_heads=12, window_size=(7,7,7), device=device,
                                  dropout_attn=dropout_attn, dropout_mlp=dropout_mlp, depth=2)
        
        self.mid1 = StageModule(in_channels=384, hidden_dimension=384,
                                  downscaling_factor=1, num_heads=12, window_size=(7,7,7), device=device,
                                  dropout_attn=0.5, dropout_mlp=0.5, mid=True, depth=2)
        
        self.mid2 = StageModule(in_channels=384, hidden_dimension=384,
                                  downscaling_factor=1, num_heads=12, window_size=(7,7,7), device=device,
                                  dropout_attn=0.5, dropout_mlp=0.5, mid=True, depth=2)
        
        self.upstage1 = CNNDouble(scale_factor=2, in_size=in_dim*4, out_size=in_dim*2, dropout=0.1)
        self.upstage2 = CNNDouble(scale_factor=2, in_size=in_dim*2, out_size=in_dim, dropout=0.1)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),
            nn.Conv3d(in_dim, 1, 3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, field, skull, tinput):

        # InstanceNorm
        field = self.patchemb1(self.norm1(field))
        skull = self.patchemb2(self.norm2(skull))

        # pressure field
        d1 = self.stage1(field) # 28
        d2 = self.stage2(d1) # 14
        d3 = self.stage3(d2) # 7

        # medical image
        s1 = self.Sstage1(skull)
        s2 = self.Sstage2(s1)
        s3 = self.Sstage3(s2)

        # tinput
        tinput = self.linear(tinput).view(-1, 1, 7, 7, 7)
        tinput = self.sa(tinput).view(-1, 7, 7, 7, 1)

        # merge
        x = d3 + s3 + tinput

        # upsample + skip connection
        x = self.upstage1(x)
        x = x + d2 + s2
        x = self.upstage2(x)
        x = x + d1 + s1
        
        x = rearrange(x, 'b d h w c -> b c d h w')

        return self.final(x)