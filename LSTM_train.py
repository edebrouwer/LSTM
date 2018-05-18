import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from custom_funs import custom_loss
from torch.autograd import Variable

from LSTM_model import Sequence, LabTestsDataset

import sys

def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print(arg)

    input_dim=2

    #Set random seed
    np.random.seed(2)
    train_dataset=LabTestsDataset(input_dim=input_dim,csv_file_serie="dummy_data.csv",csv_file_tag="dummy_death_tags.csv",file_path="./")
    #With Adam optimizer
    seq=Sequence(input_dim=input_dim)
    seq.double()
    optimizer=torch.optim.Adam(seq.parameters(), lr=0.001)
    lam=0.2
    #criterion = nn.MSELoss(size_average=False)#
    criterion = custom_loss(lam,size_average=False) #Note : for the time being, the custom loss computes the MSE and average by the total number of non NAN samples in the batch, there is no distinction of the number of non NAN samples per series.
    epochs_num=20

    dataloader = DataLoader(train_dataset, batch_size=10,shuffle=True)

    train_loss_vec=[]
    try:
        for i in range(epochs_num):
            print("EPOCH NUMBER "+str(i))
            mean_loss=0
            for i_batch, sample_batched in enumerate(dataloader): #Enumerate over the different batches in the dataset
                batch_length=sample_batched[1].size(0)
                optimizer.zero_grad()

                data_in=Variable(sample_batched[0][:,:,:-1],requires_grad=False)
                data_ref=Variable(sample_batched[0][:,:,1:],requires_grad=False)
                out = seq.fwd_test(data_in)
                mask= (data_ref == data_ref)

                #Compute Loss, backpropagate and update the weights.
                loss = criterion(data_ref[mask],out[0][:,:][mask],sample_batched[1].unsqueeze(1).double(),out[1])
                loss.backward()
                optimizer.step()

                mean_loss+=loss
            print("Loss : "+str(mean_loss/i_batch))
            train_loss_vec.append(mean_loss.item()/i_batch)
    except KeyboardInterrupt:
        torch.save(seq.state_dict(),"current_model.pt")
        torch.save(train_loss_vec,"train_loss_history.pt")
        raise

    torch.save(seq.state_dict(),"current_model.pt")
    torch.save(train_loss_vec,"train_loss_history.pt")

    to_do=input("Press t to go for test set, or any other key to abort")
    if to_do=="t":
        AUCtest=testAUC_model(seq)
        print("AUC on test set is :"+str(AUCtest))
    else:
        return(0)


def testAUC_model(mod):
    test_dataset=LabTestsDataset(input_dim=2,csv_file_serie="dummy_data_test.csv",csv_file_tag="dummy_death_tags_test.csv",file_path="./")
    dataloader_test = DataLoader(test_dataset,shuffle=True)

    true_labs=np.zeros(len(test_dataset))
    inferred_labs=np.zeros(len(test_dataset))
    for i_batch, sample_batched in enumerate(dataloader_test): #Enumerate over the different batches in the dataset
        data_in=Variable(sample_batched[0][:,:,:-1],requires_grad=False)
        data_ref=Variable(sample_batched[0][:,:,1:],requires_grad=False)
        out = mod.fwd_test(data_in)
        true_labs[i_batch]=sample_batched[1].item()
        inferred_labs[i_batch]=out[1].item()

    #Compute AUC.
    from sklearn.metrics import roc_auc_score
    AUC_test=roc_auc_score(true_labs,inferred_labs)
    #print("AUC on test samples "+str(AUC_test))
    return(AUC_test)




if __name__ == "__main__":
    main()
