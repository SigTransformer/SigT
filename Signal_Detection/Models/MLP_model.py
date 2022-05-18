import torch
from torch import nn
from torch.nn import functional as F

class MLP_SD(nn.Module):
    def __init__(self):
        super(MLP_SD,self).__init__()

        self.h_1 = nn.Linear(16384,8192)
        self.h_2 = nn.Linear(8192,4096)
        self.h_3 = nn.Linear(4096,2048)
        self.output_layer = nn.Linear(2048,2048)
        self.predict_head = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self,Y):
        Y = self.relu(self.h_1(Y))
        Y = self.relu(self.h_2(Y))
        Y = self.relu(self.h_3(Y))
        Y = self.output_layer(Y)
        X = self.predict_head(Y)
        # X = torch.where(X<0.5,torch.full_like(X,0),torch.full_like(X,1)) # used for hard max the score to original signal
                                                                           # not the knowledge we want to distill

        return X

if __name__ == '__main__':
    batch_Y = torch.randn(20,16384)
    net = MLP_SD()
    batch_X = net(batch_Y)
    print(batch_X.size(),batch_Y.size())
    print(batch_X[0,:20])
    # X = torch.where(batch_X<0.5,torch.full_like(batch_X,0),torch.full_like(batch_X,1))
    # batch_X = torch.where(batch_X>0.5,batch_X,0.0, dtype = torch.float32)
    # print(X.size())
    # print('before hard max:', batch_X[0,:20])
    # print('after hard max:', X[0,:20])



