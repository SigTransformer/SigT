import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import time


class ShapeError(Exception):
    def __init__(self,ErrorInfo):
        super(ShapeError, self).__init__()
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


class MLP_SD(nn.Module):
    def __init__(self):
        super(MLP_SD,self).__init__()

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

    def pre_process(self,src):
        if src.shape != torch.Size([256, 16384]):
            raise ShapeError('input tensor should have torch.Size([64,16384])')

        src = src.reshape(256,256,16,2,2)           # [batch_size,num_subcarriers,num_antennas,pilot&info,ReIm]
        pilot = src[:, :, :, 0, :].squeeze(dim=3).permute(0, 2, 1, 3).reshape(256, 16, -1)
        info = src[:, :, :, 1, :].squeeze(dim=3).permute(0, 2, 1, 3).reshape(256, 16, -1)

        return pilot,info

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


def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(train_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = torch.squeeze(inputs,dim=0)
        labels = torch.squeeze(labels, dim=0)
        inputs = inputs.to(device=device,dtype=torch.float32)
        labels = labels.to(device=device,dtype=torch.float32)

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
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(data_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

def evaluation(ds_type):
    eval_loader = None
    num_signals = 400 * 64
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
        labels = labels.reshape(4 * 64, 2048)

        pred = model(inputs)
        pred = torch.where(pred < 0.5, torch.full_like(pred, 0), torch.full_like(pred, 1))
        error_bits += torch.sum(torch.abs(pred-labels))

    accuracy = 1-(error_bits/(num_signals*2048))
    return accuracy


device = torch.device('cuda:1')

train_data = Signal_Data(ds_type='train',starting_channel=0,ending_channel=0,
                             num_batches=400,device=device)
val_data = Signal_Data(ds_type='val',starting_channel=0,ending_channel=0,
                             num_batches=400,device=device)
test_data = Signal_Data(ds_type='val',starting_channel=0,ending_channel=0,
                             num_batches=400,device=device)
train_loader = DataLoader(dataset=train_data,batch_size=4,shuffle=True,num_workers=16)
val_loader = DataLoader(dataset=val_data,batch_size=4,shuffle=True,num_workers=16)
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=16)

model = Transformer_for_SD(batch_size=4,sig_per_batch=64)
# model = MLP_SD()
model.load_state_dict(torch.load('./exp_state/SigT_conv_.pth'))

model = model.to(device)
model.train(False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.5)
loss_fn = nn.MSELoss()

print(evaluation('train'))
print(evaluation('val'))

# Val_ACC = torch.tensor([0.8],dtype=torch.float32,device='cpu')
#
# for epoch in tqdm(range(200)):
#     model.train(True)
#     avg_loss = train_one_epoch()
#
#     if epoch%20==19:
#         model.train(False)
#         print('Epoch {} loss: {}'.format(epoch+1,avg_loss))
#         ap = evaluation(ds_type='val')
#         print('Epoch {} acc: {}'.format(epoch+1,ap))
#         Val_ACC = torch.cat((Val_ACC,torch.tensor([ap])))
#
#
# Val_ACC = np.array(Val_ACC)

# np.savetxt('./Val_ACC_SGD2.txt',Val_ACC,delimiter='\n')
# torch.save(model.state_dict(),'./SD_SGD2.pth')