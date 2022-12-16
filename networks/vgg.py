import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19, VGG19_Weights
from misc import gram_matrix

class feat_extractor(nn.Module):
    def __init__(self, pool_layer_num = 30) -> None: # debug
        super().__init__()
        # model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1) # 'pretrained' is deprecated since 0.13
        model = vgg16(pretrained = True)
        self.features = nn.Sequential(*list(model.features.children())[:pool_layer_num])

    def forward(self, img, device = torch.device('cuda')):
        # debug
        img = img.repeat([1,3,1,1])
        self.features.eval().to(device=device)
        with torch.no_grad():
            ans = self.features(img) # [bs, 512, h // 16, w // 16]
            return ans

    def texture_loss(self, img, target, device = torch.device('cuda')):
        # G = gram_matrix(self(img, device))
        # T = gram_matrix(self(target, device))
        G = gram_matrix(img)
        T = gram_matrix(target)
        loss = nn.MSELoss()
        ans = loss(G,T)
        return ans

    def perceptual_loss(self, img, target, device = torch.device('cuda')):
        G = self(img, device)
        T = self(target, device)
        loss = nn.MSELoss()
        ans = loss(G,T)
        return ans