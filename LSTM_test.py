import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from custom_funs import custom_loss
from torch.autograd import Variable

from LSTM_model import Sequence, LabTrainDataset

import sys


def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print(arg)
    seq=load_current_model()
    print("AUC on test set: "+str(testAUC_model(seq)))

def load_current_model(): #Function to load the saved model.
    mod=Sequence(input_dim=30)
    mod.double()
    mod.cuda()
    mod.load_state_dict(torch.load("current_model.pt"))
    return(mod)


def testAUC_model(mod):
    test_dataset=LabTrainDataset(csv_file_serie="lab_short_pre_proc_test.csv")
    dataloader_test = DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=True)

    true_labs=np.zeros(len(test_dataset))
    inferred_labs=np.zeros(len(test_dataset))
    for i_batch, sample_batched in enumerate(dataloader_test): #Enumerate over the different batches in the dataset
        data_in=Variable(sample_batched[0][:,:,:-1],requires_grad=False).cuda()
        data_ref=Variable(sample_batched[0][:,:,1:],requires_grad=False).cuda()
        out = mod.fwd_test(data_in)
        true_labs=sample_batched[1].numpy()
        inferred_labs=out[1].numpy()

    #Compute AUC.
    from sklearn.metrics import roc_auc_score
    AUC_test=roc_auc_score(true_labs,inferred_labs)
    #print("AUC on test samples "+str(AUC_test))
    return(AUC_test)

if __name__ == "__main__":
    main()
