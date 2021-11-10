#!/usr/bin/env python
# coding=utf-8
import mindspore.dataset as ds
from pre_process import pre_process


TRAIN_DATA_FILE = ["RP2K_rp2k_dataset/train/RP2K_train.mindrecord"]
train_dataset = ds.MindDataset(TRAIN_DATA_FILE)
TEST_DATA_FILE = ["RP2K_rp2k_dataset/test/RP2K_test.mindrecord"]
test_dataset = ds.MindDataset(TEST_DATA_FILE)

train_dataset = pre_process(train_dataset) #数据预处理

train_it = train_dataset.create_dict_iterator(output_numpy=True)
data = next(train_it)
print(data)
#





