# -*- coding = utf-8 -*-
# @Time : 2021/11/2 2021/11/2
# @Author : moonshine
# @File : GenerateDataset.py
# @Software : PyCharm

"""
因为mindspore在训练的时候，数据集中有string类型时会报错，这个程序生成数据集时只包含data，label两个字段。
不过官方文档中在生成mindspore数据集时确实是有string字段的，所以可能有什么别的办法解决这个问题。文档中提供的图像分类例子，其数据集也确实只有两个字段，所以如果懒得去研究别的解决办法，可以原地再生成一次。
将此文件与原数据集RP2K_rp2k_dataset放在一个目录，因为之前转换的数据集和原数据集索引名相同，程序里将原数据集的文件夹名改成了RP2K_rp2k_dataset1，在程序里可以修改，修改位置在注释中标出。
 文件结构如下所示
├── GenerateDataset.py
├── RP2K_rp2k_dataset1
│   └── all
│       ├── test
│       └── train
└── RP2K_rp2k_dataset
    ├── test
    │   ├── RP2K_test.mindrecord
    │   └── RP2K_test.mindrecord.db
    └── train
        ├── RP2K_train.mindrecord
        └── RP2K_train.mindrecord.db
转换过程有一点长，请耐心等待
Tips:如果invalid path错误请创建 ./RP2K_rp2k_dataset/train 以及 ./RP2K_rp2k_dataset/test文件夹
"""

from io import BytesIO
import os

import cv2
import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter
import mindspore.dataset.vision.c_transforms as vision
from PIL import Image

MINDRECORD_FILE = ["RP2K_rp2k_dataset/test/RP2K_test.mindrecord", "RP2K_rp2k_dataset/train/RP2K_train.mindrecord"]
dic = ['test', 'train']

for a in range(0, 2):
    if os.path.exists(MINDRECORD_FILE[a]):
        os.remove(MINDRECORD_FILE[a])
        os.remove(MINDRECORD_FILE[a] + ".db")

    writer = FileWriter(file_name=MINDRECORD_FILE[a], shard_num=1)

    path1 = os.path.abspath(__file__)
    path2 = os.path.dirname(path1)

    # 这里的第二个参数是原数据集文件夹名称
    path3 = os.path.join(path2, 'RP2K_rp2k_dataset1', 'all', dic[a])
    labels = os.listdir(path3)

    nums = len(labels)

    cv_schema = {"label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema, dic[a])

    writer.add_index(["label"])

    data = []

    for l, i in enumerate(labels):
        sample = {}
        white_io = BytesIO()
        path4 = os.path.join(path3, i)
        datas = os.listdir(path4)
        # print(i)
        # k = 0
        for j in datas:
            image1 = os.path.join(path4, j)
            with open(image1, "rb") as f:
                bytes_data = f.read()
            # sample['file_name'] = j
            sample['label'] = l
            sample['data'] = bytes_data
            print(str(sample['label'] + 1) + '/' + str(nums), j)
            data.append(sample)
            writer.write_raw_data(data)
            data = []

    writer.commit()
    print(dic[a] + ' done')
