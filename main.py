#!/usr/bin/env python
# coding=utf-8
import mindspore.dataset as ds
from preprocess import preprocess
from mindspore import tensor
from lenet import LeNet5
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from model_utils.config import config
from mindspore import context
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore import Tensor

def train_lenet():
    TRAIN_DATA_FILE = ["RP2K_rp2k_dataset/train/RP2K_train.mindrecord"]
    ds_train = ds.MindDataset(TRAIN_DATA_FILE)
    TEST_DATA_FILE = ["RP2K_rp2k_dataset/test/RP2K_test.mindrecord"]
    ds_test = ds.MindDataset(TEST_DATA_FILE)

    ds_train = preprocess(ds_train)  #数据预处理
    ds_test = preprocess(ds_test)  #数据集预处理

    for data in ds_train.create_dict_iterator(output_numpy=True):
        print(data)


if __name__ == "__main__":
    train_lenet()












