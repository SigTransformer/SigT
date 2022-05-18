import argparse

import scipy.io as scio
import os
from tqdm import tqdm
import time
import numpy as np
from utils import *
import glob
import shutil

mode=0
SNRdb=25
Pilotnum=8
RX=16


data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']  # shape=?*antennas*delay*IQ           # of shape [9000,64,126,2]
# print(np.shape(x_train))
H=x_train[:,:,:,0]+1j*x_train[:,:,:,1]                          # of shape [9000,64,126]
# print(H.shape)

def generator(batch,H):
    # np.random.seed(16)
    while True:
        input_labels = []
        input_samples = []
        for row in range(0, batch):
            bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits2 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            bits3 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
            X=[bits0, bits1,bits2,bits3]
            temp = np.random.randint(0, len(H))
            HH = H[temp]
            YY = MIMO4x16(X, HH, SNRdb, mode,Pilotnum)/20 ###

            XX = np.concatenate((bits0, bits1,bits2,bits3), 0)
            input_labels.append(XX)
            input_samples.append(YY)
        batch_y = np.asarray(input_samples)                             # of shape [batch, 16384]
        batch_x = np.asarray(input_labels)                              # of shape [batch, 2048]
       # print(np.shape(batch_y))
        #print(np.shape(batch_x))
        yield (batch_y, batch_x)

def generatorXY(batch, H,  start_channel, terminate_channel):
    # np.random.seed(16)
    input_labels = []
    input_samples = []
    for row in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits2 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits3 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1, bits2, bits3]
        # temp = np.random.randint(0, len(H))
        temp = np.random.randint(start_channel,terminate_channel+1)
        HH = H[temp]
        # HH = np.mean(H[:10,:,:],axis=0)
        YY = MIMO4x16(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1, bits2, bits3), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    return batch_y, batch_x

# dataloader = DataLoader(dataset=generator(20,H),batch_size=4,shuffle=False,num_workers=2)\
# dataloader = generator(4,H)
# for i, data in enumerate(dataloader):
#     if i == 2:
#         break
#     Y,X = data
#     Y = torch.from_numpy(Y)
#     X = torch.from_numpy(X)
#     print('batch_index:',i,'Y,X\'s shape',Y.shape,X.shape)
#     print(type(Y),type(X))
    # print(X[:,:5])

class DatasetTypeError(Exception):
    def __init__(self,ErrorInfo):
        super(DatasetTypeError, self).__init__()
        self.ErrorInfo = ErrorInfo

    def __str__(self):
        return self.ErrorInfo


def Generate_Train_Data(batch_size, num_batch, root_dir, dataset_type, start_channel, terminate_channel):
    '''
    :param batch_size: size_for each batch
    :param num_batch: number of batches, the total number of data = num_batch*batch_size
    :param root_dir: directory path to store datasets (csv file)
    :param dataset_type: either train or validation or test
    :paarm start_channel: ranging from 0 to 9000
    :paarm terminate_channel: ranging from 0 to 9000, use the channel indices inclusively
    :return: save the dataset as csv files given the root directory
    '''
    if dataset_type not in ['train','validation','test']:
        raise DatasetTypeError('dataset_type must be train or validation or test!')

    # Y, X = generatorXY(batch_size, H)                 # Y.shape = [batch_size,16384]; X.shape = [batch_size,2048]
    # print(Y.shape, X.shape)

    start_time = time.time()
    for batch in tqdm(range(num_batch)):
        batch_Y, batch_X = generatorXY(batch_size, H,start_channel,terminate_channel)
        # Y = np.concatenate((Y, batch_Y), axis=0)
        # X = np.concatenate((X, batch_X), axis=0)

        # print(batch_Y.shape, batch_X.shape)                       # Y.shape = [batch_size*num_batch,16384]; X.shape = [batch_size*num_batch,2048]

        if dataset_type == 'train':
            feature_dir = root_dir+'Train_signal_'+'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//100)
            label_dir = root_dir+'Train_label_'+'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//100)
            feature_file = 'Train_signal_'+'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//100)
            label_file = 'Train_label_'+'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//100)
            if not os.path.exists(feature_dir):
                os.mkdir(feature_dir)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

            feature_path = root_dir+'Train_signal_'+'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//100)+'/{0:04d}.csv'.format(batch)
            label_path = root_dir+'Train_label_'+'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//100)+'/{0:04d}.csv'.format(batch)
        elif dataset_type == 'validation':
            feature_dir = root_dir + 'Val_signal_' + 'C{0:02d}{1:02d}_{2:02d}'.format\
                (start_channel,terminate_channel,num_batch//10)
            label_dir = root_dir + 'Val_label_' + 'C{0:02d}{1:02d}_{2:02d}'.format\
                (start_channel,terminate_channel,num_batch//10)
            feature_file = 'Val_signal_' + 'C{0:02d}{1:02d}_{2:02d}'.format\
                (start_channel,terminate_channel,num_batch//10)
            label_file = 'Val_label_' + 'C{0:02d}{1:02d}_{2:02d}'.format\
                (start_channel,terminate_channel,num_batch//10)
            if not os.path.exists(feature_dir):
                os.mkdir(feature_dir)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

            feature_path = root_dir + 'Val_signal_' + 'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//10) + '/{0:04d}.csv'.format(
                batch)
            label_path = root_dir + 'Val_label_' + 'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//10) + '/{0:04d}.csv'.format(
                batch)
            # feature_path = root_dir+'Val_signal_'+'C{}{}'.format(start_channel,terminate_channel)+'/{0:04d}.csv'.format(batch)
            # label_path = root_dir+'Val_label_'+'C{}{}'.format(start_channel,terminate_channel)+'/{0:04d}.csv'.format(batch)
        else:
            feature_dir = root_dir + 'Test_signal_' + 'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//10)
            label_dir = root_dir + 'Test_label_' + 'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//10)
            if not os.path.exists(feature_dir):
                os.mkdir(feature_dir)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

            feature_path = root_dir + 'Test_signal_' + 'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//10) + '/{0:04d}.csv'.format(
                batch)
            label_path = root_dir + 'Test_label_' + 'C{0:02d}{1:02d}_{2:02d}'.format(
                start_channel,terminate_channel,num_batch//10) + '/{0:04d}.csv'.format(
                batch)
            # feature_path = root_dir+'Test_signal_'+'C{}{}'.format(start_channel,terminate_channel)+'/{0:04d}.csv'.format(batch)
            # label_path = root_dir+'Test_label_'+'C{}{}'.format(start_channel,terminate_channel)+'/{0:04d}.csv'.format(batch)


        np.savetxt(feature_path, batch_Y, delimiter=',')
        np.savetxt(label_path, batch_X, delimiter=',')

    # merge_csv(root_dir,feature_dir,feature_file)
    # merge_csv(root_dir,label_dir,label_file)
    # shutil.rmtree(feature_dir)
    # shutil.rmtree(label_dir)
    end_time = time.time()
    print('saving the file takes',end_time-start_time,'seconds')

def merge_csv(root_dir,mid_dir,output_name):
    csv_list = glob.glob(mid_dir+'/*.csv')
    print(u'共发现%s个CSV文件' % len(csv_list))
    print(u'正在处理............')
    for i in csv_list:
        fr = open(i, 'rb').read()
        with open(root_dir+'{}.csv'.format(output_name), 'ab') as f:
            f.write(fr)
    print(u'合并完毕！')


if __name__=='__main__':
    # Generate_Train_Data(256,10,'./DataSet/','validation',99,99)
    # Generate_Train_Data(256,100,'./DataSet/','train',10)

    parser = argparse.ArgumentParser(description='Generate Train/Val/Test data')
    parser.add_argument('-BS','--batch_size', default=64, type=int, help='num_signals per batch/csv file')
    parser.add_argument('-NB','--num_batches', default=400, type=int, help='num_batches or csv files per dataset')
    parser.add_argument('-SC', '--s_channel', default=0, type=int, help='starting channel index')
    parser.add_argument('-EC', '--e_channel', default=0, type=int, help='ending channel index')
    parser.add_argument('--ds_type', default='train', type=str, help='')
    parser.add_argument('-SNR','--snr_db',default=10,help='SNR for signal in dB')
    args = parser.parse_args()
    batch_size = args.batch_size
    num_batches = args.num_batches
    s_channel = args.s_channel
    e_channel = args.e_channel
    ds_type = args.ds_type
    # snr = args.snr_db
    #
    # SNRdb = snr


    Generate_Train_Data(batch_size,num_batches,'./DataSet/',ds_type,s_channel,e_channel)
    # root_dir = './DataSet/Val_label_C0000_08'
    # merge_csv(root_dir)

