import torch.nn as nn
import numpy as np
import torch.functional as F
import torch


class selective_kernel(nn.Module):
    def __init__(self, middle, out_ch):
        super(selective_kernel, self).__init__()
        self.out_ch = out_ch
        self.middle = middle

        self.affine1 = nn.Linear(out_ch, middle)
        self.affine2 = nn.Linear(middle, out_ch)

    def forward(self, sk_conv1, sk_conv2, sk_conv3):
        sum_u = sk_conv1 + sk_conv2 + sk_conv3
        squeeze = nn.functional.adaptive_avg_pool2d(sum_u, (1, 1))
        squeeze = squeeze.view(squeeze.size(0), -1)
        z = self.affine1(squeeze)
        z = F.relu(z)
        a1 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)
        a2 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)
        a3 = self.affine2(z).reshape(-1, self.out_ch, 1, 1)

        before_softmax = torch.cat([a1, a2, a3], dim=1)
        after_softmax = F.softmax(before_softmax, dim=1)
        a1 = after_softmax[:, 0:self.out_ch, :, :]
        a1.reshape(-1, self.out_ch, 1, 1)

        a2 = after_softmax[:, self.out_ch:2*self.out_ch, :, :]
        a2.reshape(-1, self.out_ch, 1, 1)
        a3 = after_softmax[:, 2*self.out_ch:3*self.out_ch, :, :]
        a3.reshape(-1, self.out_ch, 1, 1)

        select_1 = sk_conv1 * a1
        select_2 = sk_conv2 * a2
        select_3 = sk_conv3 * a3

        return select_1 + select_2 + select_3

class RED_SK_Block(nn.Module):
    def __init__(self, out_ch=96,middle = 40):
        super(RED_SK_Block, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.conk1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.conk2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2)
        self.conk3 = nn.Conv2d(out_ch, out_ch, kernel_size=7, stride=1, padding=3)
        self.sks = selective_kernel(middle,out_ch)

    def forward(self, x):
        # encoder
        residual_1 = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        residual_2 = out
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        residual_3 = out
        out = F.relu(self.conv5(out))
        #sk
        skconv1 = F.leaky_relu(self.conk1(out))
        skconv2 = F.leaky_relu(self.conk2(out))
        skconv3 = F.leaky_relu(self.conk3(out))
        out = self.sks(skconv1,skconv2,skconv3)
        # decoder
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(F.relu(out))
        out = self.tconv3(F.relu(out))
        out += residual_2
        out = self.tconv4(F.relu(out))
        out = self.tconv5(F.relu(out))
        out += residual_1
        out = F.relu(out)
        return out
