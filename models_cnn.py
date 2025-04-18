import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_

def weights_init_cnn(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        trunc_normal_(m.weight.data, mean=0.0, std=.02)
    elif classname.find("GroupNorm") != -1:
        trunc_normal_(m.weight.data, mean=1.0, std=.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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

class Down(nn.Module):
    def __init__(self, in_size, out_size, quad=False, normalize=True, dropout=0.0):
        super(Down, self).__init__()
        if not quad: layers = [nn.Conv3d(in_size, out_size, 4, 2, 1, bias=False)]
        if quad: layers = [nn.Conv3d(in_size, out_size, 4, 4, 1, bias=False)]
        if normalize:
            layers.append(nn.GroupNorm(32, out_size))
        layers.append(nn.GELU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Mid(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Mid, self).__init__()
        layers = [
            nn.Conv3d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_size),
            nn.GELU()
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        x = self.model(x)
        return x
    
class Up(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Up, self).__init__()
        
        layers = [
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_size),
            nn.GELU()
            ]
        if dropout: layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
class CNNModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.5, skip=True):
        super(CNNModel, self).__init__()
        
        self.skip = skip # set True if using U-Net
        
        self.norm1 = nn.InstanceNorm3d(1)
        self.norm2 = nn.InstanceNorm3d(1)
        
        self.st = nn.Sequential(
            nn.Linear(343*2, 343),
            nn.GELU()
        )
        
        self.sa1 = SelfAttention(256)
        self.sa2 = SelfAttention(256)
        self.sa3 = SelfAttention(256)
        self.sa_t = SelfAttention(1, 1)
        
        self.down1 = Down(in_channels, 32) #28
        self.down2 = Down(32, 64, dropout=0.1) #14
        self.down3 = Down(64, 128, dropout=0.1) #7
        self.down4 = Down(128, 256, dropout=0.1)
        
        self.Sdown1 = Down(in_channels, 32)
        self.Sdown2 = Down(32, 64, dropout=0.1)
        self.Sdown3 = Down(64, 128, dropout=0.1)
        self.Sdown4 = Down(128, 256, dropout=0.1)
        
        self.mid1 = Mid(256, 256, dropout=dropout)
        self.mid2 = Mid(256, 256, dropout=dropout)
        
        self.up1 = Up(256, 128, dropout=0.1)
        self.up2 = Up(128, 64, dropout=0.1)
        self.up3 = Up(64, 32, dropout=0.1)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(32, out_channels, 3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, field, skull, tinput):

        field = self.norm1(field)
        skull = self.norm2(skull)

        d1 = self.down1(field)
        s1 = self.Sdown1(skull)
        
        d2 = self.down2(d1)
        s2 = self.Sdown2(s1)
        
        d3 = self.down3(d2)
        s3 = self.Sdown3(s2)

        d4 = self.sa1(self.down4(d3))
        s4 = self.sa2(self.Sdown4(s3))

        tinput = self.st(tinput).view(-1, 1, 7, 7, 7)
        tinput = self.sa_t(tinput)
        
        x = d4 + tinput + s4

        x = x + self.mid1(x) if self.skip else self.mid1(x)
        x = x + self.mid2(x) if self.skip else self.mid2(x)
        
        x = self.sa3(x)
        
        x = self.up1(x) + d3 + s3 if self.skip else self.up1(x)
        x = self.up2(x) + d2 + s2 if self.skip else self.up2(x)
        x = self.up3(x) + d1 if self.skip else self.up3(x)

        return self.final(x)