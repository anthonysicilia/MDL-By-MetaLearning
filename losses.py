
import torch

class DSCLoss:

    def __init__(self, dsc_loss_coeff):

        self.gamma = dsc_loss_coeff
        self.base = torch.nn.BCEWithLogitsLoss()
    
    def __call__(self, scores, y):

        base = self.base(scores, y)

        if self.gamma > 0:
            p = scores.sigmoid()
            p = p.view(-1, 1)
            y = y.view(1, -1)
            numer = 2 * y @ p
            denom = y.sum() + p.sum()
            dsc_loss = numer / denom
        else:
            dsc_loss = 0

        return base - dsc_loss

LOSSES = {
    'dsc-loss' : DSCLoss
}