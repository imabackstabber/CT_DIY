# adapted from https://github.com/milesial/Pytorch-UNet
import torch.nn as nn
import torch.nn.functional as F
import torch
from networks.backbone import RED_SK_Block, LayerNorm2d
from networks.attn import MDTA, CrossAttention, MDTABlock, CrossAttentionBlock

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

# adopted from https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py#L22
class SimpleGate(nn.Module):
    def __init__(self, c, FFN_Expand = 2) -> None:
        super().__init__()
        self.norm = LayerNorm2d(c)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv4(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv5(x)
        
        return x

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
            self.norm1 = LayerNorm2d(mid_channels)
            self.norm2 = LayerNorm2d(out_channels)
        else:
            self.norm1 = None
            self.norm2 = None
        
        if act == 'relu':
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
        elif act == 'gelu':
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()
        elif act == 'sg':
            # self.act1 = SimpleGate(mid_channels)
            # self.act2 = SimpleGate(out_channels)
            self.act1 = None
            self.act2 = nn.GELU()
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
                norm = 'bn' , act = 'relu', attn_mode = 'base'):
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
        if attn_mode == 'attn':
            pass
        elif attn_mode == 'sca':
            self.se = SimpleChannelAttention(64)
        elif attn_mode == 'ca':
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
        return out, x # please return the final featmap

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
            self.norm = LayerNorm2d(1)
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

class CNCL_unet(nn.Module):
    def __init__(self, noise_encoder = 'unet', content_encoder = 'unet',
                attn_mode = 'base', norm_mode = 'bn', act_mode = 'relu', 
                pre_fusion = None, fusion = 'simple') -> None:
        super().__init__()
        if noise_encoder == 'unet':
            self.noise_encoder = UNet(attn_mode = attn_mode, norm=norm_mode, act=act_mode)
        elif noise_encoder == 'sk':
            self.noise_encoder = RED_SK_Block()
        
        if content_encoder == 'unet':
            self.content_encoder = UNet(attn_mode = attn_mode, norm=norm_mode, act=act_mode)
        if fusion == 'simple':
            self.fusion_layer = SimpleFusion(pre_fusion = pre_fusion)

        # self.relu = nn.ReLU() # relu maybe needed # nope!

    def forward(self,img):
        pred_noise, _ = self.noise_encoder(img)
        pred_content, _ = self.content_encoder(img)
        pred_fusion = self.fusion_layer(pred_noise, img, pred_content)

        return {
            'pred_noise': pred_noise,
            'pred_content': pred_content,
            'pred_fusion': pred_fusion
        }

class CNCL_attn(nn.Module):
    def __init__(self, noise_encoder = 'unet', content_encoder = 'unet',
                attn_mode = 'base', norm_mode = 'bn', act_mode = 'relu',
                mdta_layer_num = 1, cross_layer_num = 1, 
                pre_fusion = None, fusion = 'simple') -> None:
        super().__init__()
        if noise_encoder == 'unet':
            self.noise_encoder = UNet(attn_mode = attn_mode, norm=norm_mode, act=act_mode)
        elif noise_encoder == 'sk':
            self.noise_encoder = RED_SK_Block()
        
        if content_encoder == 'unet':
            self.content_encoder = UNet(attn_mode = attn_mode, norm=norm_mode, act=act_mode)

        # self.noise_attn = nn.ModuleList([MDTA(dim=64) for _ in range(mdta_layer_num)])
        # self.content_attn = nn.ModuleList([MDTA(dim=64) for _ in range(mdta_layer_num)])
        # self.cross_attn = nn.ModuleList([CrossAttention(dim=64) for _ in range(cross_layer_num)])

        # use ln and residual
        # self.noise_attn = nn.ModuleList([MDTABlock(dim=64) for _ in range(mdta_layer_num)])
        # self.content_attn = nn.ModuleList([MDTABlock(dim=64) for _ in range(mdta_layer_num)])
        self.cross_attn = nn.ModuleList([CrossAttention(dim=64) for _ in range(cross_layer_num)])

        self.noise_pred = OutConv(64, 1)
        self.content_pred = OutConv(64,1)
        self.fusion_layer = SimpleFusion(pre_fusion=pre_fusion)
        
        # self.relu = nn.ReLU() # relu maybe needed # nope!

    def forward(self,img):
        _, noise_featmap = self.noise_encoder(img)
        _, content_featmap = self.content_encoder(img)

        # for layer in self.noise_attn:
        #     noise_featmap = layer(noise_featmap)
        
        # for layer in self.content_attn:
        #     content_featmap = layer(content_featmap)

        for layer in self.cross_attn:
            noise_featmap, content_featmap = layer(noise_featmap, content_featmap)

        pred_noise = self.noise_pred(noise_featmap)
        pred_content = self.content_pred(content_featmap)

        pred_fusion = self.fusion_layer(pred_noise, img, pred_content)

        return {
            'pred_noise': pred_noise,
            'pred_content': pred_content,
            'pred_fusion': pred_fusion
        }

class DualUNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, bilinear=True, 
                norm = 'bn' , act = 'relu', attn_mode = 'base', 
                cross_layer_num = 1):
        super(DualUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.content_inc = DoubleConv(in_channels, 64, norm = norm, act = act)
        self.noise_inc = DoubleConv(in_channels, 64, norm = norm, act = act)
        
        self.content_down1 = Down(64, 128, norm = norm, act = act)
        self.content_down2 = Down(128, 256, norm = norm, act = act)
        self.content_down3 = Down(256, 512, norm = norm, act = act)
        self.noise_down1 = Down(64, 128, norm = norm, act = act)
        self.noise_down2 = Down(128, 256, norm = norm, act = act)
        self.noise_down3 = Down(256, 512, norm = norm, act = act)

        factor = 2 if bilinear else 1

        self.content_down4 = Down(512, 1024 // factor, norm = norm, act = act)
        self.noise_down4 = Down(512, 1024 // factor, norm = norm, act = act)

        self.content_up1 = Up(1024, 512 // factor, bilinear, norm = norm, act = act)
        self.content_up2 = Up(512, 256 // factor, bilinear, norm = norm, act = act)
        self.content_up3 = Up(256, 128 // factor, bilinear, norm = norm, act = act)
        self.content_up4 = Up(128, 64, bilinear, norm = norm, act = act)

        self.noise_up1 = Up(1024, 512 // factor, bilinear, norm = norm, act = act)
        self.noise_up2 = Up(512, 256 // factor, bilinear, norm = norm, act = act)
        self.noise_up3 = Up(256, 128 // factor, bilinear, norm = norm, act = act)
        self.noise_up4 = Up(128, 64, bilinear, norm = norm, act = act)

        self.cross_attn1 = nn.ModuleList([CrossAttentionBlock(dim=512 // factor) for _ in range(cross_layer_num)])
        self.cross_attn2 = nn.ModuleList([CrossAttentionBlock(dim=256 // factor) for _ in range(cross_layer_num)])
        self.cross_attn3 = nn.ModuleList([CrossAttentionBlock(dim=128 // factor) for _ in range(cross_layer_num)])
        self.cross_attn4 = nn.ModuleList([CrossAttentionBlock(dim=64) for _ in range(cross_layer_num)])

        self.content_outc = OutConv(64, out_channels)
        self.noise_outc = OutConv(64, out_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: [bs * patch_n, 1, *patch_size]

        content_x1 = self.content_inc(x) # [bs * patch_n, 64, *patch_size]
        content_x2 = self.content_down1(content_x1) # [bs * patch_n, 128, *patch_size // 2]
        content_x3 = self.content_down2(content_x2) # [bs * patch_n, 256, *patch_size // 4]
        content_x4 = self.content_down3(content_x3) # [bs * patch_n, 512, *patch_size // 8]
        content_x5 = self.content_down4(content_x4) # [bs * patch_n, 512, *patch_size // 16] if bilinear

        noise_x1 = self.noise_inc(x) # [bs * patch_n, 64, *patch_size]
        noise_x2 = self.noise_down1(noise_x1) # [bs * patch_n, 128, *patch_size // 2]
        noise_x3 = self.noise_down2(noise_x2) # [bs * patch_n, 256, *patch_size // 4]
        noise_x4 = self.noise_down3(noise_x3) # [bs * patch_n, 512, *patch_size // 8]
        noise_x5 = self.noise_down4(noise_x4) # [bs * patch_n, 512, *patch_size // 16] if bilinear

        # cross-attn 1
        content_x = self.content_up1(content_x5, content_x4) # [bs * patch_n, 256, *patch_size // 8]
        noise_x = self.noise_up1(noise_x5, noise_x4) # [bs * patch_n, 256, *patch_size // 8]
        for layer in self.cross_attn1:
            content_x, noise_x = layer(content_x, noise_x)

        # cross-attn 2
        content_x = self.content_up2(content_x, content_x3) # [bs * patch_n, 128, *patch_size // 4]
        noise_x = self.noise_up2(noise_x, noise_x3) # [bs * patch_n, 128, *patch_size // 4]
        for layer in self.cross_attn2:
            content_x, noise_x = layer(content_x, noise_x)

        # cross-attn 3
        content_x = self.content_up3(content_x, content_x2) # [bs * patch_n, 256, *patch_size // 2]
        noise_x = self.noise_up3(noise_x, noise_x2) # [bs * patch_n, 256, *patch_size // 2]
        for layer in self.cross_attn3:
            content_x, noise_x = layer(content_x, noise_x)

        # cross-attn 4
        content_x = self.content_up4(content_x, content_x1) # [bs * patch_n, 64, *patch_size]
        noise_x = self.noise_up4(noise_x, noise_x1) # [bs * patch_n, 64, *patch_size]
        for layer in self.cross_attn4:
            content_x, noise_x = layer(content_x, noise_x)

        content_out = self.content_outc(content_x) # [bs * patch_n, 1, *patch_size]
        noise_out = self.noise_outc(noise_x) # [bs * patch_n, 1, *patch_size]

        return content_out, noise_out # please return the final featmap

class CNCL_full_attn(nn.Module):
    def __init__(self, attn_mode = 'base', norm_mode = 'bn', act_mode = 'relu',
                cross_layer_num = 1, 
                pre_fusion = None, fusion = 'simple') -> None:
        super().__init__()
        self.encoder = DualUNet(norm = norm_mode , act = act_mode,
                                attn_mode = attn_mode, cross_layer_num = cross_layer_num)
        self.fusion_layer = SimpleFusion(pre_fusion=pre_fusion)

    def forward(self, img):
        pred_content, pred_noise = self.encoder(img)
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
