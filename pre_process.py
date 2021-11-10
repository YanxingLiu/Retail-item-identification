import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter

decode_op = CV.Decode(rgb=True)  # jpeg解码
resize_op = CV.Resize([224, 224], Inter.BICUBIC)  # 图像resize,resize尺寸可以在这里修改


def pre_process(dataset, bs=2):
    """
    :brief: process a dataset ,include [decode, resize,shuffle,batch]
    :param dataset:  dataset to be processed
    :param bs:  batch_size
    :return: a processed dataset
    """

    dataset = dataset.map(operations=[decode_op, resize_op],
                          input_columns=["data"],
                          output_columns=["data"])
    dataset = dataset.shuffle(5)
    dataset = dataset.batch(batch_size=bs)

    return dataset
