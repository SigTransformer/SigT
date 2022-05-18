import torch
from torch import nn
import numpy as np
from Model_public4x16 import generator
import matplotlib.pyplot as plt
import scipy.io as scio
# from Models import Transformer_for_SD,MLP_SD
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import os
import argparse

class ShapeError(Exception):
    def __init__(self,ErrorInfo):
        super(ShapeError, self).__init__()
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo

class ModuleError(Exception):
    def __init__(self,ErrorInfo):
        super(ModuleError, self).__init__()
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo

class Transformer_for_SD(nn.Module):
    def __init__(self,batch_size,sig_per_batch):
        super(Transformer_for_SD, self).__init__()

        self.batch_size = batch_size
        self.sig_per_batch = sig_per_batch
        self.num_sig = self.batch_size*self.sig_per_batch
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512,nhead=4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer,num_layers=6)
        self.pool = nn.AvgPool1d(kernel_size=4,stride=4)
        self.conv = nn.Conv1d(kernel_size=4,stride=4,in_channels=512,out_channels=512)
        self.pred_head = nn.Sequential(
            # nn.LayerNorm(4096),
            nn.Linear(2048,4096),
            nn.ReLU(),
            # nn.LayerNorm(4096),
            # nn.Linear(4096, 4096),
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.Linear(4096, 2048),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.Sigmoid()
        )

    def forward(self,src):
        pilot, info = self.pre_process(src)
        if info.shape != torch.Size([self.num_sig, 16, 512]):
            raise ShapeError('info tensor should have torch.Size([16, num_signals, 512])')
        else:
            info = info.permute(1,0,2)                              # [16,num_signal,512]

        X = self.transformer(info).permute(1,2,0)                   # [num_signal,512,16]
        # X = self.pool(X).permute(0,2,1).reshape(self.num_sig,-1)    # [num_signal,512,4]-->[num_signal,2048]
        X = self.conv(X).permute(0,2,1).reshape(self.num_sig,-1)    # [num_signal,512,4]-->[num_signal,2048]
        # X = self.pool(X).permute(0,2,1)                             # [num_signal,4,512]
        out = self.pred_head(X)                                     # [num_signal,2048]

        return out

    def pre_process(self, src):
        if src.shape != torch.Size([self.batch_size,self.sig_per_batch,16384]):
            raise ShapeError('input tensor should have torch.Size([batch_size,sig_per_batch,16384])')

        src = src.reshape(self.num_sig, 256, 16, 2, 2)  # [num_signals,num_subcarriers,num_antennas,pilot&info,ReIm]
        pilot = src[:, :, :, 0, :].squeeze(dim=3).permute(0, 2, 1, 3).reshape(self.num_sig, 16, -1)
        info = src[:, :, :, 1, :].squeeze(dim=3).permute(0, 2, 1, 3).reshape(self.num_sig, 16, -1)
        # [num_signals, num_antenna, num_subcarriers*2] = [num_signals,16,512]

        return pilot, info

class LSTM(nn.Module):
    def __init__(self,batch_size,sig_per_batch):
        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.sig_per_batch = sig_per_batch
        self.num_sig = self.batch_size*self.sig_per_batch
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512,nhead=4)
        # self.transformer = nn.TransformerEncoder(self.encoder_layer,num_layers=6)
        self.lstm = nn.LSTM(input_size=512,hidden_size=256,num_layers=6,bidirectional=True)
        self.pool = nn.AvgPool1d(kernel_size=4,stride=4)
        self.conv = nn.Conv1d(kernel_size=4,stride=4,in_channels=512,out_channels=512)
        self.pred_head = nn.Sequential(
            # nn.LayerNorm(4096),
            nn.Linear(2048,4096),
            nn.ReLU(),
            # nn.LayerNorm(4096),
            # nn.Linear(4096, 4096),
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.Linear(4096, 2048),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.Sigmoid()
        )

    def forward(self,src):
        pilot, info = self.pre_process(src)
        if info.shape != torch.Size([self.num_sig, 16, 512]):
            raise ShapeError('info tensor should have torch.Size([16, num_signals, 512])')
        else:
            info = info.permute(1,0,2)                              # [16,num_signal,512]

        X = self.lstm(info)[0].permute(1,2,0)                   # [num_signal,512,16]
        # X = self.pool(X).permute(0,2,1).reshape(self.num_sig,-1)    # [num_signal,512,4]-->[num_signal,2048]
        X = self.conv(X).permute(0,2,1).reshape(self.num_sig,-1)    # [num_signal,512,4]-->[num_signal,2048]
        # X = self.pool(X).permute(0,2,1)                             # [num_signal,4,512]
        out = self.pred_head(X)                                     # [num_signal,2048]

        return out

    def pre_process(self, src):
        if src.shape != torch.Size([self.batch_size,self.sig_per_batch,16384]):
            raise ShapeError('input tensor should have torch.Size([batch_size,sig_per_batch,16384])')

        src = src.reshape(self.num_sig, 256, 16, 2, 2)  # [num_signals,num_subcarriers,num_antennas,pilot&info,ReIm]
        pilot = src[:, :, :, 0, :].squeeze(dim=3).permute(0, 2, 1, 3).reshape(self.num_sig, 16, -1)
        info = src[:, :, :, 1, :].squeeze(dim=3).permute(0, 2, 1, 3).reshape(self.num_sig, 16, -1)
        # [num_signals, num_antenna, num_subcarriers*2] = [num_signals,16,512]

        return pilot, info

class MLP_SD(nn.Module):
    def __init__(self,batch_size,sig_per_batch):
        super(MLP_SD,self).__init__()

        self.batch_size = batch_size
        self.sig_per_batch = sig_per_batch
        self.num_sig = self.batch_size * self.sig_per_batch

        self.fc_dnn1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn3 = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.Sigmoid()
        )
        self.fc_dnn4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn5 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn6 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn7 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn8 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn9 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn10 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn11 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn12 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn13 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn14 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn15 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.fc_dnn16 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

    def forward(self,src):
        pilot,H = self.pre_process(src)
        H0 = self.fc_dnn1(H[:,0,:].squeeze(dim=1))
        H1 = self.fc_dnn2(H[:,1,:].squeeze(dim=1))
        tgt = torch.cat((H0,H1),dim=1)
        H2 = self.fc_dnn3(H[:,2,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H2), dim=1)
        H3 = self.fc_dnn4(H[:,3,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H3), dim=1)
        H4 = self.fc_dnn5(H[:,4,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H4), dim=1)
        H5 = self.fc_dnn6(H[:,5,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H5), dim=1)
        H6 = self.fc_dnn7(H[:,6,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H6), dim=1)
        H7 = self.fc_dnn8(H[:,7,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H7), dim=1)
        H8 = self.fc_dnn9(H[:,8,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H8), dim=1)
        H9 = self.fc_dnn10(H[:,9,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H9), dim=1)
        H10 = self.fc_dnn11(H[:,10,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H10), dim=1)
        H11 = self.fc_dnn12(H[:,11,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H11), dim=1)
        H12 = self.fc_dnn13(H[:,12,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H12), dim=1)
        H13 = self.fc_dnn14(H[:,13,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H13), dim=1)
        H14 = self.fc_dnn15(H[:,14,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H14), dim=1)
        H15 = self.fc_dnn16(H[:,15,:].squeeze(dim=1))
        tgt = torch.cat((tgt, H15), dim=1)

        return tgt

    def pre_process(self, src):
        if src.shape != torch.Size([self.batch_size,self.sig_per_batch,16384]):
            raise ShapeError('input tensor should have torch.Size([batch_size,sig_per_batch,16384])')

        src = src.reshape(self.num_sig, 256, 16, 2, 2)  # [num_signals,num_subcarriers,num_antennas,pilot&info,ReIm]
        pilot = src[:, :, :, 0, :].squeeze(dim=3).permute(0, 2, 1, 3).reshape(self.num_sig, 16, -1)
        info = src[:, :, :, 1, :].squeeze(dim=3).permute(0, 2, 1, 3).reshape(self.num_sig, 16, -1)
        # [num_signals, num_antenna, num_subcarriers*2] = [num_signals,16,512]

        return pilot, info

class QCSI_SD(nn.Module):
    def __init__(self,batch_size,sig_per_batch):
        super(QCSI_SD,self).__init__()

        self.batch_size = batch_size
        self.sig_per_batch = sig_per_batch
        self.num_sig = self.batch_size * self.sig_per_batch
        # self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.refine_net1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        )
        self.refine_net2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        )
        self.refine_net3 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pred_head = nn.Sequential(
            # nn.LayerNorm(4096),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            # nn.LayerNorm(4096),
            # nn.Linear(4096, 4096),
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.Linear(4096, 2048),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.Sigmoid()
        )

    # def RefineNet(self, input):
    #     H = self.relu(self.conv1(input))
    #     H = self.relu(self.conv2(H))
    #     H = self.relu(self.conv3(H))
    #
    #     return H + input

    def forward(self, input):
        # H = self.preprocess_shape(input)
        _,H = self.pre_process(input)
        H = self.refine_net1(H) + H
        H = self.maxpool(H)
        H = self.refine_net2(H) + H
        H = self.maxpool(H)
        H = self.refine_net3(H) + H
        H = self.sigmoid(self.conv4(H))         # [num_sig, 2, 32,16]
        H = H.permute(0, 2, 3, 1).reshape(self.num_sig, -1)
        output = self.pred_head(H)

        return output

    def preprocess_shape(self, src):
        if src.shape != torch.Size([256, 16384]):
            raise ShapeError('input tensor should have torch.Size([64,16384])')

        src = src.reshape(256, 256, 32, 2).permute(0, 3, 1, 2)
        # sec.size() = [256,2,256,32] = [batch_size,ReIm channel,Delay,Space]
        return src

    def pre_process(self, src):
        if src.shape != torch.Size([self.batch_size,self.sig_per_batch,16384]):
            raise ShapeError('input tensor should have torch.Size([batch_size,sig_per_batch,16384])')

        src = src.reshape(self.num_sig, 256, 16, 2, 2)  # [num_signals,num_subcarriers,num_antennas,pilot&info,ReIm]
        pilot = src[:, :, :, 0, :].squeeze(dim=3).permute(0, 3, 1, 2)
        info = src[:, :, :, 1, :].squeeze(dim=3).permute(0, 3, 1, 2)
        # [num_signals, num_antenna, num_subcarriers*2] = [num_signals,16,512]

        return pilot, info

class Signal_Data(Dataset):
    def __init__(self, ds_type, starting_channel, ending_channel, num_batches, device):
        self.SC = starting_channel
        self.EC = ending_channel
        self.NB = num_batches

        if ds_type=='train':
            self.src_dir = './DataSet/Train_signal_C{0:02d}{1:02d}_{2:02d}'.format(
                self.SC,self.EC,self.NB//100)
            self.tgt_dir = './DataSet/Train_label_C{0:02d}{1:02d}_{2:02d}'.format(
                self.SC,self.EC,self.NB//100)
            # self.src_dir = './DataSet/Train_signal_C10'
            # self.tgt_dir = './DataSet/Train_label_C10'
        elif ds_type=='val':
            self.src_dir = './DataSet/Val_signal_C{0:02d}{1:02d}_{2:02d}'.format(
                self.SC,self.EC,self.NB//100)
            self.tgt_dir = './DataSet/Val_label_C{0:02d}{1:02d}_{2:02d}'.format(
                self.SC,self.EC,self.NB//100)
            # self.src_dir = './DataSet/Val_signal_C10'
            # self.tgt_dir = './DataSet/Val_label_C10'
        elif ds_type=='test':
            self.src_dir = './DataSet/Test_signal_C{0:02d}{1:02d}_{2:02d}'.format(
                self.SC,self.EC,self.NB//100)
            self.tgt_dir = './DataSet/Test_label_C{0:02d}{1:02d}_{2:02d}'.format(
                self.SC,self.EC,self.NB//100)

        self.src_list = os.listdir(self.src_dir)
        self.tgt_list = os.listdir(self.tgt_dir)


        assert len(self.src_list)==len(self.tgt_list)

        self.len = len(self.src_list)

        # self.src = torch.from_numpy(np.loadtxt(self.src_dir, delimiter=',')).to(device=device,dtype=torch.float32)
        # self.tgt = torch.from_numpy(np.loadtxt(self.tgt_dir, delimiter=',')).to(device=device,dtype=torch.float32)
        #
        # self.len = self.src.shape[0]

    def __getitem__(self, item):
        self.src_path = self.src_dir+'/'+self.src_list[item]
        self.tgt_path = self.tgt_dir+'/'+self.tgt_list[item]

        self.src = torch.from_numpy(np.loadtxt(self.src_path,delimiter=','))
        self.tgt = torch.from_numpy(np.loadtxt(self.tgt_path,delimiter=','))

        return (self.src,self.tgt)

    def __len__(self):
        return self.len

def train_one_epoch(num_iteration):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(train_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        # inputs = torch.squeeze(inputs,dim=0)
        inputs = inputs.to(device=device,dtype=torch.float32)
        labels = labels.to(device=device,dtype=torch.float32)
        labels = labels.reshape(batch_size * signal_per_batch, 2048)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % num_iteration == num_iteration-1:
            last_loss = running_loss / num_iteration # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(data_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def evaluation(ds_type):
    eval_loader = None
    num_signals = num_batches * signal_per_batch
    if ds_type=='val':
        eval_loader = val_loader
        num_signals /= 10
    elif ds_type=='train':
        eval_loader = train_loader
    # elif ds_type=='test':
    #     eval_loader = test_loader

    error_bits = 0

    for i, data in tqdm(enumerate(eval_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        # inputs = torch.squeeze(inputs, dim=0)
        # labels = labels.reshape(batch_size,2048)
        inputs = inputs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        labels = labels.reshape(batch_size * signal_per_batch, 2048)

        pred = model(inputs)
        pred = torch.where(pred < 0.5, torch.full_like(pred, 0), torch.full_like(pred, 1))
        error_bits += torch.sum(torch.abs(pred-labels))

    accuracy = 1-(error_bits/(num_signals*2048))
    return accuracy


if __name__=='__main__':
    #####################################################################################
    # argparser for easier experimenting
    parser = argparse.ArgumentParser(description='Training Config for SD')
    parser.add_argument('-BS', '--batch_size', default=10, type=int,
                        help='num batches or csv files per dataloader(64*BS)')
    parser.add_argument('-SIG', '--sig_per_batch', default=64, type=int, help='num signals per batch or csv file')
    parser.add_argument('-NW', '--num_workers', default=20, type=int, help='num_workers for dataloader')
    parser.add_argument('-NB','--num_batches',default=400,type=int,help='num iterations per epoch')
    parser.add_argument('-SC', '--s_channel', type=int, help='starting channel index', default=0)
    parser.add_argument('-EC', '--e_channel', type=int, help='ending channel index', default=0)
    parser.add_argument('-E', '--epochs', default=500, type=int)
    parser.add_argument('-LR', '--learning_rate', default=0.0001, type=float, help='learning_rate for optimizer')
    parser.add_argument('-OPTIM', '--optimizer', default='adam', type=str, help='which optimizer is used')
    parser.add_argument('-MGPU','--multi_gpus',default=False,help='whether use DataParrellel')
    parser.add_argument('-GI','--gpu_index',default=3,type=int,help='which gpu to use',required=True)
    parser.add_argument('-M','--model',default='SigT',type=str,help='model choice (SigT,fcdnn,LSTM,ComNet)')
    parser.add_argument('-LP','--log_path',required=True,type=str,help='name of file to store val_acc')
    parser.add_argument('-SP','--state_path',required=True,type=str,help='name of file to store model weights')
    # parser.add_argument('-PT','--pretrain',default=False,help='whether use pretrain model or not')

    args = parser.parse_args()


    # load argparse arguments
    starting_channel = args.s_channel
    ending_channel = args.e_channel
    batch_size = args.batch_size
    signal_per_batch = args.sig_per_batch
    num_batches = args.num_batches
    num_iter = num_batches/batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    optim_type = args.optimizer
    num_workers = args.num_workers
    multi_gpus = args.multi_gpus
    gpu_index = args.gpu_index
    model_name = args.model
    log_path = args.log_path
    state_path = args.state_path

    #####################################################################################

    device = torch.device('cuda:{}'.format(gpu_index))

    train_data = Signal_Data(ds_type='train',starting_channel=starting_channel,ending_channel=ending_channel,
                             num_batches=num_batches,device=device)
    val_data = Signal_Data(ds_type='val',starting_channel=starting_channel,ending_channel=ending_channel,
                           num_batches=num_batches,device=device)
    # test_data = Signal_Data(ds_type='test')
    train_loader = DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(dataset=val_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,
                            pin_memory=True)
    # test_loader = DataLoader(dataset=test_data,batch_size=1,shuffle=True,num_workers=16)


    # data_load_address = './data'
    # mat = scio.loadmat(data_load_address+'/Htrain.mat')
    # x_train = mat['H_train']  # shape=?*antennas*delay*IQ           # of shape [9000,64,126,2]
    # # print(np.shape(x_train))
    # H=x_train[:,:,:,0]+1j*x_train[:,:,:,1]                          # of shape [9000,64,126,2]
    # data_iter = iter(generator(64,H))

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    supported_models = ['SigT','fcdnn','csi','lstm']
    supported_optimizer = ['adam','sgd']
    if model_name not in supported_models:
        raise ModuleError('model {0} is not supported, choose between {1}'.format(model_name,supported_models))
    if model_name=='SigT':
        model = Transformer_for_SD(batch_size=batch_size,sig_per_batch=signal_per_batch)
    elif model_name=='fcdnn':
        model = MLP_SD(batch_size=batch_size,sig_per_batch=signal_per_batch)
    elif model_name=='csi':
        model = QCSI_SD(batch_size=batch_size,sig_per_batch=signal_per_batch)
    else:
        model = LSTM(batch_size=batch_size,sig_per_batch=signal_per_batch)
    # model = MLP_SD()
    # model = QCSI_SD()

    # if want to use pretrained model, uncomment the line of code below
    # model.load_state_dict(torch.load('./exp_state/SigT'))


    if optim_type not in supported_optimizer:
        raise ModuleError('optimizer {0} is not supported, choose between {1}'.format(optim_type,supported_optimizer))

    if optim_type=='sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    elif optim_type=='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # if multi_gpus:
    #
    model = model.to(device)

    Val_ACC = torch.tensor([0.5,0.5],dtype=torch.float32,device='cpu')

    for epoch in tqdm(range(num_epochs)):
        model.train(True)
        avg_loss = train_one_epoch(num_iter)
        # print('Epoch {}'.format(epoch+1))

        if epoch%20==19:
            model.train(False)
            # print('Epoch {} loss: {}'.format(epoch+1,avg_loss))
            ap = evaluation(ds_type='val')
            train_ap = evaluation(ds_type='train')
            print('Epoch {} acc: {}'.format(epoch+1,ap))
            Val_ACC = torch.cat((Val_ACC,torch.tensor([ap,train_ap])))

    Val_ACC = np.array(Val_ACC)

    np.savetxt('./exp_log/{}.txt'.format(log_path),Val_ACC,delimiter='\n')
    torch.save(model.state_dict(),'./exp_state/{}.pth'.format(state_path))





