#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd

np.random.seed(2)

class Sequence(nn.Module):
    def __init__(self,input_dim):
        print("Model Initialization")
        super(Sequence, self).__init__()
        self.input_dim=input_dim
        self.lstm1 = nn.LSTMCell(input_dim, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, input_dim)
        self.linear_out = nn.Linear(51,1)

    def fwd_test(self,data_in,future=0):

        #data_in should be sent in the Batch x Dim x Length format
        #print(data_in.shape)
        batch_dim=data_in.size(0)

        outputs = []
        h_t = torch.zeros(batch_dim, 51, dtype=torch.double).cuda()
        c_t = torch.zeros(batch_dim, 51, dtype=torch.double).cuda()
        h_t2 = torch.zeros(batch_dim, 51, dtype=torch.double).cuda()
        c_t2 = torch.zeros(batch_dim, 51, dtype=torch.double).cuda()

        #For initialization. Should be modified to set output=0 in a more simple way.
        #h_t, c_t = self.lstm1(data_in[:,0].unsqueeze(1), (h_t, c_t))
        #h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

        output = torch.zeros([batch_dim,self.input_dim],dtype=torch.double).cuda()
        #print(output)
        for i, input_t in enumerate(torch.transpose(torch.transpose(data_in,2,0),1,2)): #Data is transposed so that input_t has the batch x dim format.

            #nan_idx=np.argwhere(np.isnan(input_t.data.numpy()))[:,0] #Soon to remove
            nan_mask=(input_t!=input_t)

            input_t=input_t.clone() #Seems needed for not having in_place operation.
            #input_t[nan_idx]=output[0,nan_idx]
            #print("Input type is "+str(input_t.type()))
            #print("Output type is "+str(output.type()))
            input_t[nan_mask]=output[nan_mask]

            #print("Input type is "+str(input_t.type()))
            #print("H type is "+str(h_t.type()))
            #print("C type is "+str(c_t.type()))
            #h_t, c_t = self.lstm1(input_t.unsqueeze(0), (h_t, c_t))
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            #print(output)
            output = self.linear(h_t2)
            outputs += [output]

        lab=F.sigmoid(self.linear_out(h_t2)) #Output label.


        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        #print("FROM FUN :"+str(torch.stack(outputs, 1).size()))
        outputs = torch.transpose(torch.stack(outputs, 1),1,2) #Send the result as Batch x Dim x Length format
        #print(outputs.size())
        return [outputs,lab]

class LabTrainDataset(Dataset):
    def __init__(self,csv_file_serie="lab_short_pre_proc.csv",file_path="~/Data/MIMIC/",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        self.length=self.lab_short["HADM_ID"].nunique()
        #self.input_dim=input_dim
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        return([torch.from_numpy(self.lab_short.loc[self.lab_short["UNIQUE_ID"]==idx].as_matrix()[:,1:102]),int(self.lab_short.loc[self.lab_short["UNIQUE_ID"]==idx].as_matrix()[0,103])])

class LabTestsDataset(Dataset):
    def __init__(self,input_dim=30,csv_file_serie="lab_events_short.csv",csv_file_tag="death_tags.csv",file_path="~/Documents/Data/Full_MIMIC/",transform=None):
        self.lab_short=pd.read_csv(file_path+csv_file_serie)
        self.death_tags=pd.read_csv(file_path+csv_file_tag)
        self.length=len(self.death_tags.index)
        self.input_dim=input_dim
    def __len__(self):
        return self.length
    def __getitem__(self,idx):
        hadm_num=self.death_tags.iloc[idx]["HADM_ID"]
        death_tag=int(self.death_tags.iloc[idx]["DEATHTAG"])
        series_dat=np.empty([self.input_dim,101])
        series_dat[:]=np.nan
        for j in range(self.input_dim):
            series_dat[j,(self.lab_short.loc[(self.lab_short["HADM_ID"]==hadm_num)&(self.lab_short["LABEL_CODE"]==j),"TIME_STAMP"].as_matrix()).astype(int)]=self.lab_short.loc[(self.lab_short["HADM_ID"]==hadm_num)&(self.lab_short["LABEL_CODE"]==j),"VALUENUM"]
        return([torch.from_numpy(series_dat),death_tag])
