# adapted from https://github.com/milesial/Pytorch-UNet
import torch.nn as nn
import torch.nn.functional as F
import torch

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = nn.LayerNorm(c)
        self.norm_r = nn.LayerNorm(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class ChannelAttention(nn.Module):
    def __init__(self, in_channel = 64) -> None:
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel,in_channel // 2,1),
            nn.ReLU(),
            nn.Conv2d(in_channel // 2,in_channel,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.ca(x)

class SimpleChannelAttention(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel,in_channel,1)
        )
    
    def forward(self, x):
        return x*self.ca(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, 'In MHSA, the hidden_dim must divide num_heads'
        self.num_heads = num_heads
        self.scale_factor = (hidden_dim // num_heads) ** 0.5
        self.q = nn.Linear(input_dim, hidden_dim)
        self.k = nn.Linear(input_dim, hidden_dim)
        self.v = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, feature):
        B,L, _ = feature.shape
        q,k,v = self.q(feature), self.k(feature), self.v(feature) # [B,L,input_dim] -> [B,L,hidden_dim]
        q,k,v = q.reshape(B, L, self.num_heads, -1), \
                 k.reshape(B, L, self.num_heads, -1), \
                 v.reshape(B, L, self.num_heads, -1) # [B,L,hidden_dim] -> [B,L,N, hidden_dim_new]
        q,k,v = q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3) # [B,L,N, hidden_dim_new] -> [B,N,L, hidden_dim_new]
        attn = torch.matmul(q, k.permute(0,1,3,2)) / self.scale_factor # [B,N,L,L]
        attn_weight = F.softmax(attn, dim= -1)
        out = torch.matmul(attn_weight, v) # [B,N,L,hidden]
        out = out.permute(0,2,1,3).reshape(B,L,-1) # [B,L,N,hidden]
        out = self.out(out)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None ,norm = 'bn' , act = 'relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if norm == 'bn':
            self.norm1 = nn.BatchNorm2d(mid_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm == 'ln':
            self.norm1 = nn.LayerNorm(mid_channels)
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            self.norm1 = None
            self.norm2 = None
        
        if act == 'relu':
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
        elif act == 'gelu':
            self.act1 = nn.GELU(inplace=True)
            self.act2 = nn.GELU(inplace=True)
        else:
            self.act1 = None
            self.act2 = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        out = self.conv1(x)
        if self.norm1:
            out = self.norm1(out)
        if self.act1:
            out = self.act1(out)
        
        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)
        if self.act2:
            out = self.act2(out)
        
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels ,norm = 'bn' , act = 'relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm = norm , act = act)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm = 'bn', act = 'relu'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm, act=act)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm, act=act)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, bilinear=True, 
                norm = 'bn' , act = 'relu', mode = 'base'):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64, norm = norm, act = act)
        self.down1 = Down(64, 128, norm = norm, act = act)
        self.down2 = Down(128, 256, norm = norm, act = act)
        self.down3 = Down(256, 512, norm = norm, act = act)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm = norm, act = act)
        self.up1 = Up(1024, 512 // factor, bilinear, norm = norm, act = act)
        self.up2 = Up(512, 256 // factor, bilinear, norm = norm, act = act)
        self.up3 = Up(256, 128 // factor, bilinear, norm = norm, act = act)
        self.up4 = Up(128, 64, bilinear, norm = norm, act = act)
        if mode == 'attn':
            pass
        elif mode == 'sca':
            self.se = SimpleChannelAttention(64)
        elif mode == 'ca':
            self.se = ChannelAttention(64)
        else:
            self.se = None
        self.outc = OutConv(64, out_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [bs * patch_n, 1, *patch_size]

        x1 = self.inc(x) # [bs * patch_n, 64, *patch_size]
        x2 = self.down1(x1) # [bs * patch_n, 128, *patch_size // 2]
        x3 = self.down2(x2) # [bs * patch_n, 256, *patch_size // 4]
        x4 = self.down3(x3) # [bs * patch_n, 512, *patch_size // 8]
        x5 = self.down4(x4) # [bs * patch_n, 512, *patch_size // 16] if bilinear
        x = self.up1(x5, x4) # [bs * patch_n, 256, *patch_size // 8]
        x = self.up2(x, x3) # [bs * patch_n, 128, *patch_size // 4]
        x = self.up3(x, x2) # [bs * patch_n, 256, *patch_size // 2]
        x = self.up4(x, x1) # [bs * patch_n, 64, *patch_size]
        if self.se:
            x = self.se(x)
        out = self.outc(x) # [bs * patch_n, 1, *patch_size]
        return out

class PatchLSGAN(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchLSGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        ) # turn it into a lsgan

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, content, noisy, noise):
        img_input = torch.cat((content, noisy, noise), 1)
        return self.model(img_input)

class SimpleFusion(nn.Module):
    def __init__(self, pre_fusion = None, norm = None) -> None: # Laynorm?
        super().__init__()
        self.conv = nn.Conv2d(2,1,1)
        if pre_fusion == 'scam':
            self.pre_fusion_layer = SCAM(1)
        else:
            self.pre_fusion_layer = None
        if norm == 'ln':
            self.norm = nn.LayerNorm(1)
        else:
            self.norm = None

    def forward(self, noise, img, pred_content):
        pred_strip = img - noise
        if self.pre_fusion_layer is not None:
            pred_strip, pred_content = self.pre_fusion_layer(pred_strip, pred_content)
        out = torch.cat([pred_strip, pred_content], dim = 1) # [bs * patch_n, 1, patch_size]
        # pre-norm is needed
        if self.norm:
            out = self.norm(out)
        out = self.conv(out)
        
        return out

class CNCL(nn.Module):
    def __init__(self, noise_encoder = 'unet', content_encoder = 'unet',
                 content_encoder_mode = 'base', mode = 'base',
                 pre_fusion = None, fusion = 'simple') -> None:
        super().__init__()
        if noise_encoder == 'unet':
            self.noise_encoder = UNet(mode = mode)
        if content_encoder == 'unet':
            self.content_encoder = UNet(mode = mode)
        if fusion == 'simple':
            self.fusion_layer = SimpleFusion(pre_fusion = pre_fusion)

        # self.relu = nn.ReLU() # relu maybe needed # nope!

    def forward(self,img):
        pred_noise = self.noise_encoder(img)
        pred_content = self.content_encoder(img)
        pred_fusion = self.fusion_layer(pred_noise, img, pred_content)

        return {
            'pred_noise': pred_noise,
            'pred_content': pred_content,
            'pred_fusion': pred_fusion
        }

if __name__ == '__main__':
    shape = (4,1,64,64)
    fake = torch.randn(shape)
    unet = UNet()
    res = unet(fake)
    assert res.shape == shape, 'test fail with predicted shape {} and true shape {}'.format(res.shape, shape)
    expected_shape = (4,1,4,4)
    fakeb, reala = torch.randn(shape), torch.randn(shape)
    patchgan = PatchLSGAN()
    res = patchgan(fakeb, reala)
    assert res.shape == expected_shape, 'test fail with predicted shape {} and true shape {}'.format(res.shape, expected_shape)
