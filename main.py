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
    # context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target) #静态图模式
    context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target)
    TRAIN_DATA_FILE = ["RP2K_rp2k_dataset/train/RP2K_train.mindrecord"]
    ds_train = ds.MindDataset(TRAIN_DATA_FILE)
    TEST_DATA_FILE = ["RP2K_rp2k_dataset/test/RP2K_test.mindrecord"]
    ds_test = ds.MindDataset(TEST_DATA_FILE)
    # 数据预处理
    ds_train = preprocess(ds_train, bs=config.batch_size)
    ds_test = preprocess(ds_test, bs=config.batch_size)
    #构建网络
    network = LeNet5(num_class = config.num_classes,num_channel=3)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    #设定优化器
    net_opt = nn.Momentum(network.trainable_params(), config.lr, config.momentum)
    #定时回调函数
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=config.ckpt_path, config=config_ck)
    #设置运行环境
    if config.device_target != "Ascend":
        if config.device_target == "GPU":
            context.set_context(enable_graph_kernel=True)
        model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    else:
        model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()}, amp_level="O2")
    print("============== Starting Training ==============")
    model.train(config.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()])


if __name__ == "__main__":
    train_lenet()












