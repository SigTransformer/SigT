import torch
import numpy as np

def Accuracy(Pred,GT):
    assert Pred.shape==GT.shape
    num_signal, dim_signal = Pred.shape[0],Pred.shape[1]
    N = num_signal*dim_signal
    error_bits = torch.sum(torch.abs((Pred-GT)))
    accuracy = 1-error_bits/N

    return accuracy

if __name__=='__main__':
    pred = torch.from_numpy(np.random.binomial(n=1, p=0.5, size=(4,8)))
    gt = torch.from_numpy(np.random.binomial(n=1, p=0.5, size=(4,8)))
    print(pred,'\n',gt)
    print(Accuracy(pred,gt))

