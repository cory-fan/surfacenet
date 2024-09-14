import torch
import torch.nn.functional as F
import torch.nn as nn

class get_loss(nn.Module): #4th class for padded points 
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, weight=torch.tensor([1.0,1.3,6.4,0]).cuda()): #[0.5,1.4,3.43]
        B,_ = pred.shape
        pred = torch.concat((pred,torch.zeros(B,1).to(pred.device)),dim=-1)
        total_loss = F.nll_loss(pred, target, weight=weight)
        return total_loss
