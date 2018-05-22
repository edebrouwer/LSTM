#Create a cutom loss function
#Note : for the time being, the custom loss computes the MSE and average by the total number of non NAN samples in the batch, there is no distinction of the number of non NAN samples per series.


import torch
import torch.nn as nn
import torch.nn.functional as F

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class _Loss(nn.Module):
    def __init__(self, lam=0,size_average=True, reduce=True):
        super(_Loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.lam=lam

class custom_loss(_Loss):
    def __init__(self,lam,size_average=True,reduce=True):
        super(custom_loss, self).__init__(size_average,reduce)
        self.lam=lam
    def forward(self,target_series,est_series,target_label,est_label):
        _assert_no_grad(target_series)
        #_assert_no_grad(target_label)
        loss_series=F.mse_loss(est_series, target_series, size_average=self.size_average, reduce=self.reduce)
        loss_label=F.binary_cross_entropy(est_label, target_label,size_average=self.size_average,reduce=self.reduce)
        total_loss=loss_series+self.lam*loss_label
        #print("SIZE AVERAGE = "+str(self.size_average))
        #print("Series LOSS"+str(loss_series))
        #print(loss_label)
        return total_loss
