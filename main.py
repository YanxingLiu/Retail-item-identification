#!/usr/bin/env python
# coding=utf-8
import mindspore.dataset as ds
from preprocess import preprocess
from mindspore import Model

TRAIN_DATA_FILE = ["RP2K_rp2k_dataset/train/RP2K_train.mindrecord"]
train_dataset = ds.MindDataset(TRAIN_DATA_FILE)
TEST_DATA_FILE = ["RP2K_rp2k_dataset/test/RP2K_test.mindrecord"]
test_dataset = ds.MindDataset(TEST_DATA_FILE)

train_dataset = preprocess(train_dataset) #数据预处理
test_dataset = preprocess(test_dataset) #数据集预处理








