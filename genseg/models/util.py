import torch
import torch.nn.functional as F
import torch.nn as nn

class get_loss(nn.Module): #4th class for padded points 
    def __init__(self,dataset):
        super(get_loss, self).__init__()
        if (dataset=="kumamoto"):
            self.weight=torch.tensor([1, 1.9, 4.3, 0]).cuda()
        elif (dataset=="haiti"):
            self.weight=torch.tensor([1, 1.04736973, 10.89780135,0]).cuda()
        else:
            raise NotImplementedError 
    def forward(self, pred, target): #[0.5,1.4,3.43]
        B,_ = pred.shape
        pred = torch.concat((pred,torch.zeros(B,1).to(pred.device)),dim=-1)
        total_loss = F.nll_loss(pred, target, weight=self.weight)
        return total_loss
