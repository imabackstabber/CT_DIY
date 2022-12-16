import torch
import torch.nn as nn

# adopted from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def gram_matrix(img):
    a, b, c, d = img.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = img.view(a , b, c * d)  # resise F_XL into \hat F_XL
    features_t = img.view(a , b, c * d).permute(0,2,1)  # resise F_XL into \hat F_XL

    G = torch.bmm(features, features_t)  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.

    return G.div(a * b * c * d)

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count