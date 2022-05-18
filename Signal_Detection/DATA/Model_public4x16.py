import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import *
import struct

mode=0
SNRdb=10
Pilotnum=8
# ###########################以下仅为信道数据载入和链路使用范例############
RX=16
import scipy.io as scio

data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']  # shape=?*antennas*delay*IQ           # of shape [9000,64,126,2]
# print(np.shape(x_train))
H=x_train[:,:,:,0]+1j*x_train[:,:,:,1]                          # of shape [9000,64,126,2]
#
# model = keras.MyModel() #定义自己的模型
# model.summary()

####################使用链路和信道数据产生训练数据##########
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
#####训练#########
# model.fit_generator(generator(1000,H),steps_per_epoch=2,epochs=10)
# model.save_weights('TrainMIMO/model_4x16.h5')

######产生模拟验证环境########
def generatorXY(batch, H):
    # np.random.seed(16)
    input_labels = []
    input_samples = []
    for row in range(0, batch):
        bits0 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits1 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits2 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        bits3 = np.random.binomial(n=1, p=0.5, size=(128 * 4,))
        X = [bits0, bits1, bits2, bits3]
        temp = np.random.randint(0, len(H))
        HH = H[temp]
        YY = MIMO4x16(X, HH, SNRdb, mode, Pilotnum) / 20  ###
        XX = np.concatenate((bits0, bits1, bits2, bits3), 0)
        input_labels.append(XX)
        input_samples.append(YY)
    batch_y = np.asarray(input_samples)
    batch_x = np.asarray(input_labels)
    return batch_y, batch_x

#############产生Y与X用于验证模型性能，非评测用
# Y, X = generatorXY(100, H)
# X = np.array(np.floor(X + 0.5), dtype=np.bool)
# print(np.shape(Y))
# print(np.shape(X))

##########X与Y的存储方式
# np.savetxt('Yval.csv', Y, delimiter=',')
# X.tofile('Xval.bin')

############load model and predict Xval_pre from Yval
# Y_1 = np.loadtxt('Yval.csv', dtype=np.float, delimiter=',')
# model.load_weights('TrainMIMO/model_4x16.h5')
# Xval_pre = model.predict(Y_1)
# Xval_pre = np.array(np.floor(Xval_pre + 0.5), dtype=np.bool)
# Xval_pre.tofile('Xval_pre.bin')

##############性能指标
# print('score=',100*np.mean(X==Xval_pre))